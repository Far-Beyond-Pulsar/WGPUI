//! The native HTTP compatibility boundary.
//!
//! `wgpui-core` stores application services but deliberately knows nothing
//! about HTTP. This crate supplies the native implementation boundary and
//! re-exports the legacy request vocabulary so applications can migrate their
//! configured clients without introducing a second HTTP protocol.

use std::sync::Arc;

use wgpui_core::App;

pub use gpui_http_client as http_client;
pub use gpui_http_client::http;
pub use gpui_http_client::{
    AsyncBody, AsyncBody as HttpBody, Builder, FollowRedirects, HttpClient, HttpClientWithProxy,
    HttpClientWithUrl, HttpRequestExt, Inner, Method, RedirectPolicy, Request, Response, Result,
    StatusCode, Uri, Url,
};

/// The client type accepted by native application configuration.
pub type BoxedHttpClient = Arc<dyn HttpClient>;

/// A client that makes the absence of application HTTP configuration explicit.
pub struct NullHttpClient;

impl HttpClient for NullHttpClient {
    fn type_name(&self) -> &'static str {
        "NullHttpClient"
    }

    fn user_agent(&self) -> Option<&http_client::http::HeaderValue> {
        None
    }

    fn send(
        &self,
        _request: Request<AsyncBody>,
    ) -> futures::future::BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
        Box::pin(async { Err(anyhow::anyhow!("No HttpClient available")) })
    }

    fn proxy(&self) -> Option<&Url> {
        None
    }
}

/// The application-owned wrapper stored in core's type-erased service map.
#[derive(Clone)]
pub struct HttpClientService {
    client: BoxedHttpClient,
}

impl HttpClientService {
    pub fn new(client: BoxedHttpClient) -> Self {
        Self { client }
    }

    pub fn client(&self) -> BoxedHttpClient {
        Arc::clone(&self.client)
    }
}

/// Configure and access the native HTTP client without adding HTTP to core's
/// dependency graph.
pub trait AppHttpClientExt {
    fn with_http_client(self, client: BoxedHttpClient) -> Self
    where
        Self: Sized;

    fn set_http_client(&mut self, client: BoxedHttpClient);

    fn http_client(&self) -> BoxedHttpClient;

    fn configured_http_client(&self) -> Option<BoxedHttpClient>;

    fn install_default_http_client(&mut self);
}

impl AppHttpClientExt for App {
    fn with_http_client(mut self, client: BoxedHttpClient) -> Self {
        self.set_http_client(client);
        self
    }

    fn set_http_client(&mut self, client: BoxedHttpClient) {
        self.set_global(HttpClientService::new(client));
    }

    fn http_client(&self) -> BoxedHttpClient {
        self.configured_http_client()
            .unwrap_or_else(|| Arc::new(NullHttpClient))
    }

    fn configured_http_client(&self) -> Option<BoxedHttpClient> {
        self.global::<HttpClientService>()
            .map(|service| service.client())
    }

    fn install_default_http_client(&mut self) {
        if self.configured_http_client().is_none() {
            self.set_http_client(Arc::new(NullHttpClient));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::future::BoxFuture;
    use futures::io::AsyncReadExt;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct RecordingClient {
        request: std::sync::Mutex<Option<Request<AsyncBody>>>,
        response: std::sync::Mutex<Option<anyhow::Result<Response<AsyncBody>>>>,
        user_agent: Option<http_client::http::HeaderValue>,
    }

    impl HttpClient for RecordingClient {
        fn type_name(&self) -> &'static str {
            "RecordingClient"
        }

        fn user_agent(&self) -> Option<&http_client::http::HeaderValue> {
            self.user_agent.as_ref()
        }

        fn send(
            &self,
            request: Request<AsyncBody>,
        ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
            *self.request.lock().expect("request lock") = Some(request);
            let response = self
                .response
                .lock()
                .expect("response lock")
                .take()
                .expect("response configured");
            Box::pin(async move { response })
        }

        fn proxy(&self) -> Option<&Url> {
            None
        }
    }

    struct PendingRequest {
        dropped: Arc<AtomicBool>,
    }

    impl std::future::Future for PendingRequest {
        type Output = anyhow::Result<Response<AsyncBody>>;

        fn poll(
            self: std::pin::Pin<&mut Self>,
            _context: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            std::task::Poll::Pending
        }
    }

    impl Drop for PendingRequest {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    struct PendingClient {
        dropped: Arc<AtomicBool>,
    }

    impl HttpClient for PendingClient {
        fn type_name(&self) -> &'static str {
            "PendingClient"
        }

        fn user_agent(&self) -> Option<&http_client::http::HeaderValue> {
            None
        }

        fn send(
            &self,
            _request: Request<AsyncBody>,
        ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
            Box::pin(PendingRequest {
                dropped: Arc::clone(&self.dropped),
            })
        }

        fn proxy(&self) -> Option<&Url> {
            None
        }
    }

    #[test]
    fn app_keeps_configured_client_outside_core() {
        let mut app = App::new();
        let client: BoxedHttpClient = Arc::new(NullHttpClient);
        app.set_http_client(Arc::clone(&client));
        assert_eq!(app.http_client().type_name(), "NullHttpClient");
        assert!(app.configured_http_client().is_some());
    }

    #[test]
    fn get_helper_encodes_redirect_policy_for_a_fake_client() {
        let client = Arc::new(RecordingClient {
            request: std::sync::Mutex::new(None),
            response: std::sync::Mutex::new(Some(Ok(Response::builder()
                .status(200)
                .body(AsyncBody::empty())
                .expect("response")))),
            user_agent: Some(http_client::http::HeaderValue::from_static("test-agent")),
        });
        let client_trait: Arc<dyn HttpClient> = Arc::clone(&client) as _;
        futures::executor::block_on(client_trait.get(
            "http://test.example/redirect",
            AsyncBody::empty(),
            true,
        ))
        .expect("fake response");
        assert_eq!(
            client
                .request
                .lock()
                .expect("request lock")
                .as_ref()
                .and_then(|request| request.extensions().get::<RedirectPolicy>()),
            Some(&RedirectPolicy::FollowAll)
        );
    }

    #[test]
    fn fake_client_propagates_transport_failures() {
        let client: BoxedHttpClient = Arc::new(RecordingClient {
            request: std::sync::Mutex::new(None),
            response: std::sync::Mutex::new(Some(Err(anyhow::anyhow!("transport down")))),
            user_agent: None,
        });
        let result = futures::executor::block_on(client.get(
            "http://test.example/failure",
            AsyncBody::empty(),
            true,
        ));
        assert!(result.is_err());
        assert_eq!(
            result.err().map(|error| error.to_string()).as_deref(),
            Some("transport down")
        );
    }

    #[test]
    fn configured_accessors_preserve_user_agent_and_proxy_metadata() {
        let client = Arc::new(RecordingClient {
            request: std::sync::Mutex::new(None),
            response: std::sync::Mutex::new(Some(Ok(Response::builder()
                .status(200)
                .body(AsyncBody::empty())
                .expect("response")))),
            user_agent: Some(http_client::http::HeaderValue::from_static("test-agent")),
        });
        let proxy = Url::parse("http://proxy.example:8080").expect("proxy URL");
        let configured: BoxedHttpClient =
            Arc::new(HttpClientWithProxy::new_url(client, Some(proxy.clone())));
        let mut app = App::new();
        app.set_http_client(configured);
        let selected = app.http_client();
        assert_eq!(
            selected.user_agent().and_then(|value| value.to_str().ok()),
            Some("test-agent")
        );
        assert_eq!(selected.proxy(), Some(&proxy));
    }

    #[test]
    fn streaming_body_can_be_consumed_without_a_concrete_http_client() {
        let mut body = HttpBody::from_bytes(Bytes::from_static(b"streamed"));
        let bytes = futures::executor::block_on(async {
            let mut output = Vec::new();
            body.read_to_end(&mut output).await.expect("body reads");
            output
        });
        assert_eq!(bytes, b"streamed");
    }

    #[test]
    fn task_cancellation_cancels_a_pending_request_future() {
        let dropped = Arc::new(AtomicBool::new(false));
        let client: BoxedHttpClient = Arc::new(PendingClient {
            dropped: Arc::clone(&dropped),
        });
        let app = App::new();
        let mut task = app.spawn(async move {
            client
                .get("http://test.example/pending", AsyncBody::empty(), true)
                .await
        });
        app.run_pending_tasks();
        task.cancel();
        assert!(task.is_cancelled());
        app.run_pending_tasks();
        assert!(matches!(
            futures::executor::block_on(&mut task),
            Err(wgpui_core::TaskError::Cancelled)
        ));
        assert!(dropped.load(Ordering::SeqCst));
    }
}
