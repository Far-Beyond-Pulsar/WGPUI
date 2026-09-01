//! Asynchronous resource loading for native image elements.
//!
//! Resolution is kept beside the image decoder rather than in `wgpui-core`:
//! core owns app state and tasks, while this layer owns resource identity and
//! the decision to use the configured native HTTP client for URI resources.

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;

use futures::AsyncReadExt;
use wgpui_core::{App, Task};
use wgpui_http_client::{AppHttpClientExt, AsyncBody, BoxedHttpClient, StatusCode};
use wgpui_text::shaping::SharedString;

use crate::image_cache::{DecodedImage, ImageDecodeError, decode_async};

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

impl ImageAssetLoader {
    /// Resolve and decode a resource on the app's background executor.
    pub fn load(
        source: impl Into<Resource>,
        app: &App,
        asset_source: Arc<dyn AssetSource>,
    ) -> Task<Result<DecodedImage, AssetLoadError>> {
        let client = app.http_client();
        let source = source.into();
        app.background_spawn(async move {
            let bytes = Self::load_bytes(source, client, asset_source).await?;
            decode_async(bytes).await.map_err(AssetLoadError::Decode)
        })
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
