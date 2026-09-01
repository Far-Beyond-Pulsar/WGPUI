//! Asynchronous resource loading for native image elements.
//!
//! Resolution is kept beside the image decoder rather than in `wgpui-core`:
//! core owns app state and tasks, while this layer owns resource identity and
//! the decision to use the configured native HTTP client for URI resources.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures::AsyncReadExt;
use wgpui_core::{App, TaskError};
use wgpui_http_client::{AppHttpClientExt, AsyncBody, BoxedHttpClient, StatusCode};
use wgpui_text::shaping::SharedString;

use crate::image_cache::{DecodedImage, ImageDecodeError, decode_async};

type RedrawCallback = Arc<dyn Fn() + Send + Sync>;
type RedrawCallbacks = Arc<Mutex<HashMap<u64, RedrawCallback>>>;

/// A typed, cancellable application asset operation.
pub trait Asset: 'static {
    type Source: Clone + Hash + Send + Sync + 'static;
    type Output: Send + 'static;

    fn load(
        source: Self::Source,
        app: &mut App,
    ) -> impl std::future::Future<Output = Self::Output> + Send + 'static;
}

/// Adds diagnostics and task ownership to an [`Asset`] load.
pub struct AssetLogger<A>(PhantomData<A>);

impl<A: Asset> AssetLogger<A> {
    pub fn load(
        source: A::Source,
        app: &mut App,
    ) -> impl std::future::Future<Output = A::Output> + Send + 'static {
        A::load(source, app)
    }
}

/// The public image representation returned by the compatibility loader.
pub type RenderImage = DecodedImage;

/// Errors from an application asset operation.
#[derive(Debug)]
pub enum ImageCacheError {
    Load(AssetLoadError),
    Loading,
    Failed(String),
    Cancelled(TaskError),
    Other(anyhow::Error),
}

impl std::fmt::Display for ImageCacheError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(error) => error.fmt(formatter),
            Self::Loading => formatter.write_str("asset is still loading"),
            Self::Failed(error) => write!(formatter, "asset load failed: {error}"),
            Self::Cancelled(error) => error.fmt(formatter),
            Self::Other(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ImageCacheError {}
impl From<AssetLoadError> for ImageCacheError {
    fn from(error: AssetLoadError) -> Self {
        Self::Load(error)
    }
}
impl From<anyhow::Error> for ImageCacheError {
    fn from(error: anyhow::Error) -> Self {
        Self::Other(error)
    }
}

/// Delay used by legacy image loading examples before a loading replacement is shown.
pub const LOADING_DELAY: std::time::Duration = std::time::Duration::from_millis(250);

/// Application-owned source and decoded-image registry.
pub struct AssetRegistry {
    source: Arc<dyn AssetSource>,
    entries: Arc<Mutex<HashMap<Resource, RegistryEntry>>>,
    redraw_callbacks: RedrawCallbacks,
    next_callback: AtomicU64,
}

#[derive(Clone)]
struct RegistryEntry {
    generation: u64,
    state: AssetState,
    image: Option<Arc<RenderImage>>,
}

/// The observable state of a registry entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssetState {
    Loading,
    Ready,
    Failed(String),
    Cancelled,
}

/// A subscription used by a renderer to request a targeted redraw when an
/// asset changes state.
pub struct AssetRedrawSubscription {
    callbacks: RedrawCallbacks,
    id: u64,
}

/// A cancellable registry-owned load.
pub struct AssetRequest {
    registry: Arc<Mutex<HashMap<Resource, RegistryEntry>>>,
    redraw_callbacks: RedrawCallbacks,
    resource: Resource,
    generation: u64,
    task: Option<wgpui_core::Task<()>>,
    detached: bool,
}

impl AssetRegistry {
    pub fn new(source: Arc<dyn AssetSource>) -> Self {
        Self {
            source,
            entries: Arc::new(Mutex::new(HashMap::new())),
            redraw_callbacks: Arc::new(Mutex::new(HashMap::new())),
            next_callback: AtomicU64::new(0),
        }
    }

    pub fn source(&self) -> &Arc<dyn AssetSource> {
        &self.source
    }

    pub fn invalidate(&self, resource: &Resource) -> bool {
        let removed = match self.entries.lock() {
            Ok(mut entries) => entries.remove(resource).is_some(),
            Err(poisoned) => poisoned.into_inner().remove(resource).is_some(),
        };
        if removed {
            self.notify_redraw();
        }
        removed
    }

    pub fn cached(&self, resource: &Resource) -> Option<Arc<RenderImage>> {
        match self.entries.lock() {
            Ok(entries) => entries.get(resource).and_then(|entry| entry.image.clone()),
            Err(poisoned) => poisoned
                .into_inner()
                .get(resource)
                .and_then(|entry| entry.image.clone()),
        }
    }

    /// Return the current state without doing any I/O or decoding.
    pub fn state(&self, resource: &Resource) -> Option<AssetState> {
        match self.entries.lock() {
            Ok(entries) => entries.get(resource).map(|entry| entry.state.clone()),
            Err(poisoned) => poisoned
                .into_inner()
                .get(resource)
                .map(|entry| entry.state.clone()),
        }
    }

    /// Return a cached image. A miss is reported as a stateful error; this
    /// method never starts work and never blocks the caller.
    pub fn load_cached(&self, resource: Resource) -> Result<Arc<RenderImage>, ImageCacheError> {
        let entry = match self.entries.lock() {
            Ok(entries) => entries.get(&resource).cloned(),
            Err(poisoned) => poisoned.into_inner().get(&resource).cloned(),
        };
        match entry {
            Some(RegistryEntry {
                state: AssetState::Ready,
                image: Some(image),
                ..
            }) => Ok(image),
            Some(RegistryEntry {
                state: AssetState::Loading,
                ..
            })
            | None => Err(ImageCacheError::Loading),
            Some(RegistryEntry {
                state: AssetState::Failed(error),
                ..
            }) => Err(ImageCacheError::Failed(error)),
            Some(RegistryEntry {
                state: AssetState::Cancelled,
                ..
            }) => Err(ImageCacheError::Cancelled(TaskError::Cancelled)),
            Some(RegistryEntry {
                state: AssetState::Ready,
                image: None,
                ..
            }) => Err(ImageCacheError::Other(anyhow::anyhow!(
                "ready asset has no decoded image"
            ))),
        }
    }

    /// Start a deduplicated load using the application's configured HTTP
    /// client. The returned handle can cancel work that is still pending.
    pub fn load(&self, resource: Resource, app: &App) -> AssetRequest {
        self.load_async(resource, app)
    }

    /// Start a deduplicated background load without doing any work on the
    /// render thread.
    pub fn load_async(&self, resource: Resource, app: &App) -> AssetRequest {
        let (generation, should_start) = match self.entries.lock() {
            Ok(mut entries) => match entries.get(&resource) {
                Some(entry) => (entry.generation, false),
                None => {
                    entries.insert(
                        resource.clone(),
                        RegistryEntry {
                            generation: 1,
                            state: AssetState::Loading,
                            image: None,
                        },
                    );
                    (1, true)
                }
            },
            Err(poisoned) => {
                let mut entries = poisoned.into_inner();
                let generation = entries
                    .get(&resource)
                    .map_or(1, |entry| entry.generation.wrapping_add(1).max(1));
                entries.insert(
                    resource.clone(),
                    RegistryEntry {
                        generation,
                        state: AssetState::Loading,
                        image: None,
                    },
                );
                (generation, true)
            }
        };

        if should_start {
            let entries = Arc::clone(&self.entries);
            let source = Arc::clone(&self.source);
            let redraw_callbacks = Arc::clone(&self.redraw_callbacks);
            let client = app.configured_http_client();
            let load_resource = resource.clone();
            let task = app.background_spawn(async move {
                let result = ImageAssetLoader::load_with_optional_client(
                    load_resource.clone(),
                    client,
                    source,
                )
                .await
                .map(Arc::new);
                let mut notify = false;
                match entries.lock() {
                    Ok(mut entries) => {
                        if let Some(entry) = entries.get_mut(&load_resource)
                            && entry.generation == generation
                            && entry.state == AssetState::Loading
                        {
                            notify = true;
                            match result {
                                Ok(image) => {
                                    entry.image = Some(image);
                                    entry.state = AssetState::Ready;
                                }
                                Err(error) => {
                                    entry.image = None;
                                    entry.state = AssetState::Failed(error.to_string());
                                }
                            }
                        }
                    }
                    Err(poisoned) => {
                        let mut entries = poisoned.into_inner();
                        if let Some(entry) = entries.get_mut(&load_resource)
                            && entry.generation == generation
                            && entry.state == AssetState::Loading
                        {
                            notify = true;
                            match result {
                                Ok(image) => {
                                    entry.image = Some(image);
                                    entry.state = AssetState::Ready;
                                }
                                Err(error) => {
                                    entry.image = None;
                                    entry.state = AssetState::Failed(error.to_string());
                                }
                            }
                        }
                    }
                }
                if notify {
                    let callbacks = match redraw_callbacks.lock() {
                        Ok(callbacks) => callbacks.values().cloned().collect::<Vec<_>>(),
                        Err(poisoned) => {
                            poisoned.into_inner().values().cloned().collect::<Vec<_>>()
                        }
                    };
                    for callback in callbacks {
                        callback();
                    }
                }
            });
            return AssetRequest {
                registry: Arc::clone(&self.entries),
                redraw_callbacks: Arc::clone(&self.redraw_callbacks),
                resource,
                generation,
                task: Some(task),
                detached: false,
            };
        }

        AssetRequest {
            registry: Arc::clone(&self.entries),
            redraw_callbacks: Arc::clone(&self.redraw_callbacks),
            resource,
            generation,
            task: None,
            detached: false,
        }
    }

    /// Register a renderer callback for targeted redraw invalidation.
    pub fn subscribe_redraw(
        &self,
        callback: impl Fn() + Send + Sync + 'static,
    ) -> AssetRedrawSubscription {
        let id = self
            .next_callback
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1);
        match self.redraw_callbacks.lock() {
            Ok(mut callbacks) => {
                callbacks.insert(id, Arc::new(callback));
            }
            Err(poisoned) => {
                poisoned.into_inner().insert(id, Arc::new(callback));
            }
        }
        AssetRedrawSubscription {
            callbacks: Arc::clone(&self.redraw_callbacks),
            id,
        }
    }

    fn notify_redraw(&self) {
        let callbacks = match self.redraw_callbacks.lock() {
            Ok(callbacks) => callbacks.values().cloned().collect::<Vec<_>>(),
            Err(poisoned) => poisoned.into_inner().values().cloned().collect::<Vec<_>>(),
        };
        for callback in callbacks {
            callback();
        }
    }
}

impl AssetRequest {
    pub fn state(&self) -> Option<AssetState> {
        match self.registry.lock() {
            Ok(entries) => entries.get(&self.resource).map(|entry| entry.state.clone()),
            Err(poisoned) => poisoned
                .into_inner()
                .get(&self.resource)
                .map(|entry| entry.state.clone()),
        }
    }

    pub fn cancel(&mut self) {
        let cancelled = match self.registry.lock() {
            Ok(mut entries) => {
                let Some(entry) = entries.get_mut(&self.resource) else {
                    return;
                };
                if entry.generation != self.generation || entry.state != AssetState::Loading {
                    return;
                }
                entry.state = AssetState::Cancelled;
                true
            }
            Err(poisoned) => {
                let mut entries = poisoned.into_inner();
                let Some(entry) = entries.get_mut(&self.resource) else {
                    return;
                };
                if entry.generation != self.generation || entry.state != AssetState::Loading {
                    return;
                }
                entry.state = AssetState::Cancelled;
                true
            }
        };
        if cancelled {
            if let Some(task) = self.task.as_mut() {
                task.cancel();
            }
            notify_callbacks(&self.redraw_callbacks);
        }
    }

    /// Let the registry-owned operation finish after this handle is dropped.
    pub fn detach(mut self) {
        self.detached = true;
        if let Some(task) = self.task.take() {
            task.detach();
        }
    }
}

impl Drop for AssetRedrawSubscription {
    fn drop(&mut self) {
        match self.callbacks.lock() {
            Ok(mut callbacks) => {
                callbacks.remove(&self.id);
            }
            Err(poisoned) => {
                poisoned.into_inner().remove(&self.id);
            }
        }
    }
}

impl Drop for AssetRequest {
    fn drop(&mut self) {
        if self.task.is_some() && !self.detached && self.state() == Some(AssetState::Loading) {
            self.cancel();
        }
    }
}

fn notify_callbacks(callbacks: &RedrawCallbacks) {
    let callbacks = match callbacks.lock() {
        Ok(callbacks) => callbacks.values().cloned().collect::<Vec<_>>(),
        Err(poisoned) => poisoned.into_inner().values().cloned().collect::<Vec<_>>(),
    };
    for callback in callbacks {
        callback();
    }
}

impl Default for AssetRegistry {
    fn default() -> Self {
        Self::new(Arc::new(()))
    }
}

/// A source of embedded assets supplied by an application.
pub trait AssetSource: 'static + Send + Sync {
    fn load(&self, path: &str) -> anyhow::Result<Option<Cow<'_, [u8]>>>;

    fn list(&self, path: &str) -> anyhow::Result<Vec<SharedString>>;
}

impl AssetSource for () {
    fn load(&self, _path: &str) -> anyhow::Result<Option<Cow<'_, [u8]>>> {
        Ok(None)
    }

    fn list(&self, _path: &str) -> anyhow::Result<Vec<SharedString>> {
        Ok(Vec::new())
    }
}

/// A resource accepted by the native image loader.
pub type ImageSource = Resource;

/// A cheaply cloneable URI spelling used by the compatibility examples.
pub type SharedUri = String;

/// The three resource locations supported by the native loader.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Resource {
    Uri(String),
    Path(PathBuf),
    Embedded(String),
}

impl From<String> for Resource {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

impl From<&str> for Resource {
    fn from(value: &str) -> Self {
        if wgpui_http_client::Uri::try_from(value).is_ok() {
            Self::Uri(value.to_string())
        } else {
            Self::Embedded(value.to_string())
        }
    }
}

impl From<PathBuf> for Resource {
    fn from(value: PathBuf) -> Self {
        Self::Path(value)
    }
}

impl From<Arc<std::path::Path>> for Resource {
    fn from(value: Arc<std::path::Path>) -> Self {
        Self::Path(value.as_ref().to_path_buf())
    }
}

impl From<SharedString> for Resource {
    fn from(value: SharedString) -> Self {
        Self::from(value.as_str())
    }
}

/// Errors raised while resolving and decoding a resource.
#[derive(Debug)]
pub enum AssetLoadError {
    Http(anyhow::Error),
    MissingHttpClient,
    Io(std::io::Error),
    Embedded(anyhow::Error),
    BadStatus {
        uri: String,
        status: StatusCode,
        body: String,
    },
    Decode(ImageDecodeError),
}

impl std::fmt::Display for AssetLoadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http(error) => write!(formatter, "HTTP request failed: {error}"),
            Self::MissingHttpClient => {
                formatter.write_str("no HTTP client is configured for URI asset loading")
            }
            Self::Io(error) => write!(formatter, "asset I/O failed: {error}"),
            Self::Embedded(error) => write!(formatter, "embedded asset failed: {error}"),
            Self::BadStatus { uri, status, body } => {
                write!(
                    formatter,
                    "unexpected HTTP status for {uri}: {status}, body: {body}"
                )
            }
            Self::Decode(error) => write!(formatter, "image decode failed: {error}"),
        }
    }
}

impl std::error::Error for AssetLoadError {}

/// The native equivalent of the legacy image asset loader.
pub struct ImageAssetLoader;

impl Asset for ImageAssetLoader {
    type Source = Resource;
    type Output = Result<Arc<RenderImage>, ImageCacheError>;

    fn load(
        source: Self::Source,
        app: &mut App,
    ) -> impl std::future::Future<Output = Self::Output> + Send + 'static {
        let client = app.configured_http_client();
        let assets: Arc<dyn AssetSource> = app.global::<AssetRegistry>().map_or_else(
            || Arc::new(()) as Arc<dyn AssetSource>,
            |registry| registry.source().clone(),
        );
        async move {
            let bytes = ImageAssetLoader::load_bytes(source, client, assets).await?;
            Ok(Arc::new(
                decode_async(bytes).await.map_err(AssetLoadError::Decode)?,
            ))
        }
    }
}

pub struct ImgResourceLoader;
impl Asset for ImgResourceLoader {
    type Source = Resource;
    type Output = Result<Arc<RenderImage>, ImageCacheError>;
    fn load(
        source: Self::Source,
        app: &mut App,
    ) -> impl std::future::Future<Output = Self::Output> + Send + 'static {
        <ImageAssetLoader as Asset>::load(source, app)
    }
}

impl ImageAssetLoader {
    /// Resolve and decode a resource on the app's background executor.
    pub fn load(
        source: impl Into<Resource>,
        app: &App,
        asset_source: Arc<dyn AssetSource>,
    ) -> impl std::future::Future<Output = Result<Arc<RenderImage>, ImageCacheError>> {
        let client = app.configured_http_client();
        let source = source.into();
        let task = app.background_spawn(async move {
            let bytes = Self::load_bytes(source, client, asset_source).await?;
            decode_async(bytes)
                .await
                .map_err(AssetLoadError::Decode)
                .map(Arc::new)
        });
        async move {
            let result = task.await.map_err(ImageCacheError::Cancelled)?;
            result.map_err(ImageCacheError::from)
        }
    }

    /// Resolve and decode a resource with an explicitly selected client.
    ///
    /// This is useful for tests and for callers that keep resource loading in
    /// a separate executor while still using the same configured-client seam.
    pub async fn load_with_client(
        source: impl Into<Resource>,
        client: BoxedHttpClient,
        asset_source: Arc<dyn AssetSource>,
    ) -> Result<DecodedImage, AssetLoadError> {
        let bytes = Self::load_bytes(source.into(), Some(client), asset_source).await?;
        decode_async(bytes).await.map_err(AssetLoadError::Decode)
    }

    async fn load_with_optional_client(
        source: impl Into<Resource>,
        client: Option<BoxedHttpClient>,
        asset_source: Arc<dyn AssetSource>,
    ) -> Result<DecodedImage, AssetLoadError> {
        let bytes = Self::load_bytes(source.into(), client, asset_source).await?;
        decode_async(bytes).await.map_err(AssetLoadError::Decode)
    }

    async fn load_bytes(
        source: Resource,
        client: Option<BoxedHttpClient>,
        asset_source: Arc<dyn AssetSource>,
    ) -> Result<Vec<u8>, AssetLoadError> {
        match source {
            Resource::Path(path) => std::fs::read(path).map_err(AssetLoadError::Io),
            Resource::Uri(uri) => {
                let Some(client) = client else {
                    return Err(AssetLoadError::MissingHttpClient);
                };
                let mut response = client
                    .get(&uri, AsyncBody::empty(), true)
                    .await
                    .map_err(AssetLoadError::Http)?;
                let status = response.status();
                let mut body = Vec::new();
                response
                    .body_mut()
                    .read_to_end(&mut body)
                    .await
                    .map_err(AssetLoadError::Io)?;
                if !status.is_success() {
                    let mut text = String::from_utf8_lossy(&body).into_owned();
                    let first_line = text.lines().next().unwrap_or("").trim_end();
                    text.truncate(first_line.len());
                    return Err(AssetLoadError::BadStatus {
                        uri,
                        status,
                        body: text,
                    });
                }
                Ok(body)
            }
            Resource::Embedded(path) => asset_source
                .load(&path)
                .map_err(AssetLoadError::Embedded)?
                .map(Cow::into_owned)
                .ok_or_else(|| {
                    AssetLoadError::Embedded(anyhow::anyhow!("embedded resource not found: {path}"))
                }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future::BoxFuture;
    use std::sync::Mutex;
    use wgpui_http_client::{HttpClient, Request, Response};

    struct MemoryAssets;
    impl AssetSource for MemoryAssets {
        fn load(&self, path: &str) -> anyhow::Result<Option<Cow<'_, [u8]>>> {
            Ok((path == "icon.svg").then_some(Cow::Borrowed(br#"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="2"><rect width="4" height="2" fill="red"/></svg>"#)))
        }
        fn list(&self, _path: &str) -> anyhow::Result<Vec<SharedString>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn registry_decodes_caches_and_invalidates_application_assets() {
        let registry = AssetRegistry::new(Arc::new(MemoryAssets));
        let resource = Resource::Embedded("icon.svg".into());
        let app = App::create();
        registry.load_async(resource.clone(), &app).detach();
        wait_for_state(&registry, &resource, AssetState::Ready);
        let first = registry.load_cached(resource.clone()).expect("asset loads");
        let second = registry.load_cached(resource.clone()).expect("cache hit");
        assert!(Arc::ptr_eq(&first, &second));
        assert!(registry.invalidate(&resource));
        assert!(registry.cached(&resource).is_none());
        let missing = Resource::Embedded("missing.svg".into());
        registry.load_async(missing.clone(), &app).detach();
        wait_for_state(&registry, &missing, AssetState::Failed(String::new()));
        assert!(matches!(
            registry.load_cached(missing),
            Err(ImageCacheError::Failed(_))
        ));
    }

    fn wait_for_state(registry: &AssetRegistry, resource: &Resource, expected: AssetState) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            if let Some(state) = registry.state(resource)
                && (state == expected
                    || matches!(
                        (&state, &expected),
                        (AssetState::Failed(_), AssetState::Failed(_))
                    ))
            {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "asset did not reach {expected:?}"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn uri_failure_without_a_configured_client_is_retained_as_a_failure_state() {
        let registry = AssetRegistry::new(Arc::new(()));
        let app = App::create();
        let resource = Resource::Uri("https://example.test/image.png".into());
        registry.load_async(resource.clone(), &app).detach();
        wait_for_state(&registry, &resource, AssetState::Failed(String::new()));
        let Some(AssetState::Failed(error)) = registry.state(&resource) else {
            panic!("URI request did not retain its failure state");
        };
        assert!(error.contains("no HTTP client is configured"));
    }

    struct FakeClient {
        response: Mutex<Option<anyhow::Result<Response<AsyncBody>>>>,
    }

    impl HttpClient for FakeClient {
        fn type_name(&self) -> &'static str {
            "FakeClient"
        }

        fn user_agent(&self) -> Option<&wgpui_http_client::http_client::http::HeaderValue> {
            None
        }

        fn send(
            &self,
            _request: Request<AsyncBody>,
        ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
            let result = self
                .response
                .lock()
                .expect("fake response lock")
                .take()
                .expect("fake request only once");
            Box::pin(async move { result })
        }

        fn proxy(&self) -> Option<&wgpui_http_client::Url> {
            None
        }
    }

    #[test]
    fn failed_status_preserves_only_the_first_response_line() {
        let client: BoxedHttpClient = Arc::new(FakeClient {
            response: Mutex::new(Some(Ok(Response::builder()
                .status(404)
                .body(AsyncBody::from("not found\nsecret details"))
                .expect("response")))),
        });
        let result = futures::executor::block_on(ImageAssetLoader::load_with_client(
            Resource::Uri("https://example.test/image".into()),
            client,
            Arc::new(()),
        ));
        match result {
            Err(AssetLoadError::BadStatus { status, body, .. }) => {
                assert_eq!(status, StatusCode::NOT_FOUND);
                assert_eq!(body, "not found");
            }
            other => panic!("expected status error, got {other:?}"),
        }
    }

    #[test]
    fn registry_uses_the_configured_client_for_uri_resources() {
        let client: BoxedHttpClient = Arc::new(FakeClient {
            response: Mutex::new(Some(Ok(Response::builder()
                .status(200)
                .body(AsyncBody::from(br#"<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2"/></svg>"#.to_vec()))
                .expect("response")))),
        });
        let mut app = App::create();
        app.set_http_client(client);
        let registry = AssetRegistry::new(Arc::new(()));
        let resource = Resource::Uri("https://example.test/image.svg".into());
        registry.load_async(resource.clone(), &app).detach();
        wait_for_state(&registry, &resource, AssetState::Ready);
        assert!(registry.cached(&resource).is_some());
    }

    #[test]
    fn registry_notifies_redraw_subscribers_after_a_state_transition() {
        let redraws = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let registry = AssetRegistry::new(Arc::new(MemoryAssets));
        let subscription_counter = Arc::clone(&redraws);
        let _subscription = registry.subscribe_redraw(move || {
            subscription_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        let app = App::create();
        let resource = Resource::Embedded("icon.svg".into());
        registry.load_async(resource.clone(), &app).detach();
        wait_for_state(&registry, &resource, AssetState::Ready);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while redraws.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "redraw callback was not called"
            );
            std::thread::yield_now();
        }
        assert_eq!(redraws.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn uncached_access_does_not_block_and_reports_loading() {
        let registry = AssetRegistry::new(Arc::new(MemoryAssets));
        let resource = Resource::Embedded("icon.svg".into());
        let started = std::time::Instant::now();
        assert!(matches!(
            registry.load_cached(resource),
            Err(ImageCacheError::Loading)
        ));
        assert!(started.elapsed() < std::time::Duration::from_millis(50));
    }

    #[test]
    fn a_pending_request_can_be_cancelled_without_committing_a_late_result() {
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let client: BoxedHttpClient = Arc::new(PendingClient {
            dropped: Arc::clone(&dropped),
            started: Arc::clone(&started),
        });
        let mut app = App::create();
        app.set_http_client(client);
        let registry = AssetRegistry::new(Arc::new(()));
        let resource = Resource::Uri("https://example.test/pending".into());
        let mut request = registry.load_async(resource.clone(), &app);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !started.load(std::sync::atomic::Ordering::SeqCst) {
            assert!(
                std::time::Instant::now() < deadline,
                "pending request did not start"
            );
            std::thread::yield_now();
        }
        let observer = registry.load_async(resource.clone(), &app);
        drop(observer);
        assert_eq!(registry.state(&resource), Some(AssetState::Loading));
        request.cancel();
        assert_eq!(request.state(), Some(AssetState::Cancelled));
        drop(request);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !dropped.load(std::sync::atomic::Ordering::SeqCst) {
            assert!(
                std::time::Instant::now() < deadline,
                "pending request was not cancelled"
            );
            std::thread::yield_now();
        }
        assert_eq!(registry.state(&resource), Some(AssetState::Cancelled));
    }

    struct PendingClient {
        dropped: Arc<std::sync::atomic::AtomicBool>,
        started: Arc<std::sync::atomic::AtomicBool>,
    }

    impl HttpClient for PendingClient {
        fn type_name(&self) -> &'static str {
            "PendingClient"
        }

        fn user_agent(&self) -> Option<&wgpui_http_client::http_client::http::HeaderValue> {
            None
        }

        fn send(
            &self,
            _request: Request<AsyncBody>,
        ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
            self.started
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Box::pin(PendingResponse {
                dropped: Arc::clone(&self.dropped),
            })
        }

        fn proxy(&self) -> Option<&wgpui_http_client::Url> {
            None
        }
    }

    struct PendingResponse {
        dropped: Arc<std::sync::atomic::AtomicBool>,
    }

    impl std::future::Future for PendingResponse {
        type Output = anyhow::Result<Response<AsyncBody>>;

        fn poll(
            self: std::pin::Pin<&mut Self>,
            _context: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            std::task::Poll::Pending
        }
    }

    impl Drop for PendingResponse {
        fn drop(&mut self) {
            self.dropped
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
}
