//! Asynchronous resource loading for native image elements.
//!
//! Resolution is kept beside the image decoder rather than in `wgpui-core`:
//! core owns app state and tasks, while this layer owns resource identity and
//! the decision to use the configured native HTTP client for URI resources.

use std::borrow::Cow;
use std::path::PathBuf;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use futures::AsyncReadExt;
use wgpui_core::{App, TaskError};
use wgpui_http_client::{AppHttpClientExt, AsyncBody, BoxedHttpClient, StatusCode};
use wgpui_text::shaping::SharedString;

use crate::image_cache::{DecodedImage, ImageDecodeError, decode_async};

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
    pub fn load(source: A::Source, app: &mut App) -> impl std::future::Future<Output = A::Output> {
        A::load(source, app)
    }
}

/// The public image representation returned by the compatibility loader.
pub type RenderImage = DecodedImage;

/// Errors from an application asset operation.
#[derive(Debug)]
pub enum ImageCacheError {
    Load(AssetLoadError),
    Cancelled(TaskError),
    Other(anyhow::Error),
}

impl std::fmt::Display for ImageCacheError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(error) => error.fmt(formatter),
            Self::Cancelled(error) => error.fmt(formatter),
            Self::Other(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ImageCacheError {}
impl From<AssetLoadError> for ImageCacheError {
    fn from(error: AssetLoadError) -> Self { Self::Load(error) }
}
impl From<anyhow::Error> for ImageCacheError {
    fn from(error: anyhow::Error) -> Self { Self::Other(error) }
}

/// Delay used by legacy image loading examples before a loading replacement is shown.
pub const LOADING_DELAY: std::time::Duration = std::time::Duration::from_millis(250);

/// Application-owned source and decoded-image registry.
pub struct AssetRegistry {
    source: Arc<dyn AssetSource>,
    entries: Mutex<HashMap<Resource, RegistryEntry>>,
}

#[derive(Clone)]
struct RegistryEntry {
    generation: u64,
    image: Arc<RenderImage>,
}

impl AssetRegistry {
    pub fn new(source: Arc<dyn AssetSource>) -> Self {
        Self { source, entries: Mutex::new(HashMap::new()) }
    }

    pub fn source(&self) -> &Arc<dyn AssetSource> { &self.source }

    pub fn invalidate(&self, resource: &Resource) -> bool {
        self.entries.lock().map(|mut entries| entries.remove(resource).is_some()).unwrap_or(false)
    }

    pub fn cached(&self, resource: &Resource) -> Option<Arc<RenderImage>> {
        self.entries.lock().ok()?.get(resource).map(|entry| Arc::clone(&entry.image))
    }

    pub fn load_cached(&self, resource: Resource) -> Result<Arc<RenderImage>, ImageCacheError> {
        if let Some(image) = self.cached(&resource) { return Ok(image); }
        let image = futures::executor::block_on(ImageAssetLoader::load_with_client(
            resource.clone(),
            Arc::new(wgpui_http_client::NullHttpClient),
            Arc::clone(&self.source),
        ))?;
        let image = Arc::new(image);
        self.entries.lock().map_err(|_| anyhow::anyhow!("asset registry lock poisoned"))?
            .insert(resource, RegistryEntry { generation: 0, image: Arc::clone(&image) });
        Ok(image)
    }
}

impl Default for AssetRegistry {
    fn default() -> Self { Self::new(Arc::new(())) }
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

    fn load(source: Self::Source, app: &mut App) -> impl std::future::Future<Output = Self::Output> + Send + 'static {
        let client = app.http_client();
        let assets: Arc<dyn AssetSource> = app.global::<AssetRegistry>()
            .map_or_else(|| Arc::new(()) as Arc<dyn AssetSource>, |registry| registry.source().clone());
        async move {
            let bytes = ImageAssetLoader::load_bytes(source, client, assets).await?;
            Ok(Arc::new(decode_async(bytes).await.map_err(AssetLoadError::Decode)?))
        }
    }
}

pub struct ImgResourceLoader;
impl Asset for ImgResourceLoader {
    type Source = Resource;
    type Output = Result<Arc<RenderImage>, ImageCacheError>;
    fn load(source: Self::Source, app: &mut App) -> impl std::future::Future<Output = Self::Output> + Send + 'static {
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
        let client = app.http_client();
        let source = source.into();
        let task = app.background_spawn(async move {
            let bytes = Self::load_bytes(source, client, asset_source).await?;
            decode_async(bytes).await.map_err(AssetLoadError::Decode).map(Arc::new)
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
        let bytes = Self::load_bytes(source.into(), client, asset_source).await?;
        decode_async(bytes).await.map_err(AssetLoadError::Decode)
    }

    async fn load_bytes(
        source: Resource,
        client: BoxedHttpClient,
        asset_source: Arc<dyn AssetSource>,
    ) -> Result<Vec<u8>, AssetLoadError> {
        match source {
            Resource::Path(path) => std::fs::read(path).map_err(AssetLoadError::Io),
            Resource::Uri(uri) => {
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
        fn list(&self, _path: &str) -> anyhow::Result<Vec<SharedString>> { Ok(Vec::new()) }
    }

    #[test]
    fn registry_decodes_caches_and_invalidates_application_assets() {
        let registry = AssetRegistry::new(Arc::new(MemoryAssets));
        let resource = Resource::Embedded("icon.svg".into());
        let first = registry.load_cached(resource.clone()).expect("asset loads");
        let second = registry.load_cached(resource.clone()).expect("cache hit");
        assert!(Arc::ptr_eq(&first, &second));
        assert!(registry.invalidate(&resource));
        assert!(registry.cached(&resource).is_none());
        assert!(registry.load_cached(Resource::Embedded("missing.svg".into())).is_err());
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
}
