//! Optional loopback transport for frozen captures.
//!
//! The transport is deliberately started by an application, never by the
//! renderer. A per-server cookie is required in the first message, and the
//! listener is restricted to IPv4 loopback so an endpoint cannot be exposed
//! accidentally on a LAN interface.

use crate::capture::{CaptureError, CaptureService};
use crate::protocol::{
    Capabilities, Capability, ClientMessage, ErrorCode, ProtocolError, Request, Response,
    SUPPORTED_PROTOCOL_VERSION, ServerMessage, read_message, write_message,
};
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const AUTH_TOKEN_LENGTH: usize = 32;
const DEFAULT_CONNECTION_TIMEOUT: Duration = Duration::from_millis(250);

#[derive(Clone)]
pub struct Endpoint {
    address: SocketAddr,
    auth_token: [u8; AUTH_TOKEN_LENGTH],
}

impl std::fmt::Debug for Endpoint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Endpoint")
            .field("address", &self.address)
            .field("auth_token", &"<redacted>")
            .finish()
    }
}

impl Endpoint {
    pub fn address(&self) -> SocketAddr {
        self.address
    }
    pub fn auth_token(&self) -> &[u8; AUTH_TOKEN_LENGTH] {
        &self.auth_token
    }
}

#[derive(Debug, Clone)]
pub struct LocalIpcConfig {
    pub bind_address: IpAddr,
    pub port: u16,
    pub max_message_bytes: usize,
    pub max_connections: usize,
    pub connection_timeout: Duration,
    pub capabilities: Capabilities,
}

impl Default for LocalIpcConfig {
    fn default() -> Self {
        Self {
            bind_address: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 0,
            max_message_bytes: crate::protocol::DEFAULT_MAX_MESSAGE_BYTES,
            max_connections: 4,
            connection_timeout: DEFAULT_CONNECTION_TIMEOUT,
            capabilities: Capabilities::ALL,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LocalIpcError {
    #[error("IPC endpoints must bind to 127.0.0.1")]
    NonLoopbackAddress,
    #[error("IPC message limit must be between 1 and {maximum} bytes")]
    InvalidMessageLimit { maximum: usize },
    #[error("IPC connection limit must be at least one")]
    InvalidConnectionLimit,
    #[error("failed to create an authentication cookie: {0}")]
    Random(String),
    #[error("failed to bind the local IPC endpoint: {0}")]
    Bind(#[source] io::Error),
    #[error("failed to configure the local IPC endpoint: {0}")]
    Configure(#[source] io::Error),
    #[error("failed to start the local IPC worker: {0}")]
    Thread(#[source] io::Error),
    #[error("the local IPC worker panicked")]
    WorkerPanicked,
}

pub struct LocalIpcServer {
    endpoint: Endpoint,
    shutdown: Arc<AtomicBool>,
    active_connections: Arc<AtomicUsize>,
    worker: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for LocalIpcServer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalIpcServer")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

impl LocalIpcServer {
    pub fn start(
        service: Arc<dyn CaptureService>,
        config: LocalIpcConfig,
    ) -> Result<Self, LocalIpcError> {
        if config.max_message_bytes == 0 || config.max_message_bytes > u32::MAX as usize {
            return Err(LocalIpcError::InvalidMessageLimit {
                maximum: u32::MAX as usize,
            });
        }
        if config.max_connections == 0 {
            return Err(LocalIpcError::InvalidConnectionLimit);
        }
        if !config.bind_address.is_loopback() {
            return Err(LocalIpcError::NonLoopbackAddress);
        }
        let listener = TcpListener::bind(SocketAddr::new(config.bind_address, config.port))
            .map_err(LocalIpcError::Bind)?;
        listener
            .set_nonblocking(true)
            .map_err(LocalIpcError::Configure)?;
        let mut auth_token = [0; AUTH_TOKEN_LENGTH];
        getrandom::fill(&mut auth_token)
            .map_err(|error| LocalIpcError::Random(error.to_string()))?;
        let endpoint = Endpoint {
            address: listener.local_addr().map_err(LocalIpcError::Configure)?,
            auth_token,
        };
        let shutdown = Arc::new(AtomicBool::new(false));
        let active_connections = Arc::new(AtomicUsize::new(0));
        let worker_shutdown = shutdown.clone();
        let worker_connections = active_connections.clone();
        let worker_endpoint = endpoint.clone();
        let worker = thread::Builder::new()
            .name("wgpui-devtools-ipc".into())
            .spawn(move || {
                accept_connections(
                    listener,
                    service,
                    config,
                    worker_endpoint,
                    worker_shutdown,
                    worker_connections,
                );
            })
            .map_err(LocalIpcError::Thread)?;
        Ok(Self {
            endpoint,
            shutdown,
            active_connections,
            worker: Some(worker),
        })
    }

    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    pub fn active_connections(&self) -> usize {
        self.active_connections.load(Ordering::Acquire)
    }

    pub fn shutdown(&mut self) -> Result<(), LocalIpcError> {
        self.shutdown.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| LocalIpcError::WorkerPanicked)?;
        }
        Ok(())
    }
}

impl Drop for LocalIpcServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take()
            && let Err(error) = worker.join()
        {
            eprintln!("wgpui-devtools IPC worker panicked during drop: {error:?}");
        }
    }
}

fn accept_connections(
    listener: TcpListener,
    service: Arc<dyn CaptureService>,
    config: LocalIpcConfig,
    endpoint: Endpoint,
    shutdown: Arc<AtomicBool>,
    active_connections: Arc<AtomicUsize>,
) {
    let mut child_workers = Vec::new();
    while !shutdown.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((stream, _)) => {
                if active_connections.fetch_add(1, Ordering::AcqRel) >= config.max_connections {
                    active_connections.fetch_sub(1, Ordering::AcqRel);
                    continue;
                }
                let connection_service = service.clone();
                let connection_config = config.clone();
                let connection_endpoint = endpoint.clone();
                let connection_shutdown = shutdown.clone();
                let connection_count = active_connections.clone();
                let child_worker = thread::spawn(move || {
                    handle_connection(
                        stream,
                        connection_service,
                        connection_config,
                        connection_endpoint,
                        connection_shutdown,
                    );
                    connection_count.fetch_sub(1, Ordering::AcqRel);
                });
                child_workers.push(child_worker);
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(2))
            }
            Err(_) => break,
        }
    }
    for child_worker in child_workers {
        if child_worker.join().is_err() {
            eprintln!("wgpui-devtools IPC connection worker panicked");
        }
    }
}

fn handle_connection(
    mut stream: TcpStream,
    service: Arc<dyn CaptureService>,
    config: LocalIpcConfig,
    endpoint: Endpoint,
    shutdown: Arc<AtomicBool>,
) {
    if stream.set_nonblocking(false).is_err()
        || stream
            .set_read_timeout(Some(config.connection_timeout))
            .is_err()
        || stream
            .set_write_timeout(Some(config.connection_timeout))
            .is_err()
    {
        return;
    }
    let hello = match read_message::<ClientMessage>(&mut stream, config.max_message_bytes) {
        Ok(ClientMessage::Hello {
            protocol_version,
            capabilities,
            max_message_bytes,
            auth_token,
        }) => (
            protocol_version,
            capabilities,
            max_message_bytes,
            auth_token,
        ),
        Ok(ClientMessage::Request(_)) => {
            send_error(
                &mut stream,
                config.max_message_bytes,
                ErrorCode::AuthenticationFailed,
                "hello is required before requests",
            );
            return;
        }
        Err(_) => return,
    };
    let (protocol_version, client_capabilities, client_maximum, auth_token) = hello;
    if !constant_time_equal(&auth_token, &endpoint.auth_token) {
        send_error(
            &mut stream,
            config.max_message_bytes,
            ErrorCode::AuthenticationFailed,
            "authentication failed",
        );
        return;
    }
    if protocol_version != SUPPORTED_PROTOCOL_VERSION {
        send_error(
            &mut stream,
            config.max_message_bytes,
            ErrorCode::UnsupportedVersion,
            "unsupported protocol version",
        );
        return;
    }
    let negotiated_maximum = usize::try_from(client_maximum)
        .ok()
        .filter(|maximum| *maximum > 0)
        .map_or(config.max_message_bytes, |maximum| {
            maximum.min(config.max_message_bytes)
        });
    let capabilities = config.capabilities.intersection(client_capabilities);
    if write_message(
        &mut stream,
        &ServerMessage::Hello {
            protocol_version: SUPPORTED_PROTOCOL_VERSION,
            capabilities,
            max_message_bytes: negotiated_maximum as u32,
        },
        negotiated_maximum,
    )
    .is_err()
    {
        return;
    }
    while !shutdown.load(Ordering::Acquire) {
        let message = match read_message::<ClientMessage>(&mut stream, negotiated_maximum) {
            Ok(message) => message,
            Err(ProtocolError::Io(error))
                if matches!(
                    error.kind(),
                    io::ErrorKind::UnexpectedEof
                        | io::ErrorKind::ConnectionReset
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::WouldBlock
                ) =>
            {
                return;
            }
            Err(_) => {
                send_error(
                    &mut stream,
                    negotiated_maximum,
                    ErrorCode::InvalidRequest,
                    "invalid message",
                );
                return;
            }
        };
        let ClientMessage::Request(request) = message else {
            send_error(
                &mut stream,
                negotiated_maximum,
                ErrorCode::InvalidRequest,
                "hello may only be sent once",
            );
            return;
        };
        let Some(required_capability) = request_capability(&request) else {
            send_error(
                &mut stream,
                negotiated_maximum,
                ErrorCode::InvalidRequest,
                "unknown request",
            );
            continue;
        };
        if !capabilities.contains(required_capability) {
            send_error(
                &mut stream,
                negotiated_maximum,
                ErrorCode::CapabilityUnavailable,
                "capability was not negotiated",
            );
            continue;
        }
        let response = match request {
            Request::ArmCapture => service.arm_capture().map(|_| Response::Accepted),
            Request::StopCapture => service.stop_capture().map(|_| Response::Accepted),
            Request::Snapshot => service.snapshot().map(Response::Capture),
            Request::ReadResource { id, offset, length } => usize::try_from(length)
                .map_err(|_| CaptureError::InvalidReadbackRange)
                .and_then(|length| service.read_resource(id, offset, length))
                .map(Response::Resource),
        };
        match response {
            Ok(response) => {
                if let Err(error) = write_message(
                    &mut stream,
                    &ServerMessage::Response(response),
                    negotiated_maximum,
                ) {
                    if matches!(error, ProtocolError::MessageTooLarge { .. }) {
                        send_error(
                            &mut stream,
                            negotiated_maximum,
                            ErrorCode::MessageTooLarge,
                            "response exceeds the negotiated message limit",
                        );
                    }
                    return;
                }
            }
            Err(error) => send_error(
                &mut stream,
                negotiated_maximum,
                error_code(&error),
                &error.to_string(),
            ),
        }
    }
}

fn request_capability(request: &Request) -> Option<Capability> {
    Some(match request {
        Request::ArmCapture => Capability::CaptureArm,
        Request::StopCapture => Capability::CaptureStop,
        Request::Snapshot => Capability::Snapshot,
        Request::ReadResource { .. } => Capability::ResourceReadback,
    })
}

fn error_code(error: &CaptureError) -> ErrorCode {
    match error {
        CaptureError::UnknownResource(_)
        | CaptureError::InvalidReadbackRange
        | CaptureError::ReadbackTooLarge { .. } => ErrorCode::ResourceUnavailable,
        CaptureError::InvalidState => ErrorCode::CaptureUnavailable,
        CaptureError::CaptureTooLarge { .. } | CaptureError::Serialization(_) => {
            ErrorCode::MessageTooLarge
        }
        CaptureError::DuplicateResource
        | CaptureError::UnsupportedSchemaVersion(_)
        | CaptureError::InvalidFile
        | CaptureError::Io(_) => ErrorCode::Internal,
    }
}

fn send_error(stream: &mut TcpStream, maximum_bytes: usize, code: ErrorCode, message: &str) {
    let response = ServerMessage::Error {
        code,
        message: message.to_owned(),
    };
    if let Err(error) = write_message(stream, &response, maximum_bytes) {
        eprintln!("failed to write IPC error response: {error}");
    }
}

fn constant_time_equal(left: &[u8; AUTH_TOKEN_LENGTH], right: &[u8; AUTH_TOKEN_LENGTH]) -> bool {
    let mut difference = 0u8;
    for (left_byte, right_byte) in left.iter().zip(right.iter()) {
        difference |= left_byte ^ right_byte;
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{
        CaptureController, FrozenCapture, ResourceId, ResourceKind, ResourceSnapshot,
    };
    use crate::protocol::{ClientMessage, Request, ServerMessage, decode_message, encode_message};
    use std::net::TcpStream;
    use std::sync::Arc;

    fn connected_server() -> (LocalIpcServer, Arc<CaptureController>) {
        let controller = Arc::new(CaptureController::new());
        controller.arm().expect("test capture should arm");
        controller.begin_collection();
        controller
            .publish_frozen(
                FrozenCapture::with_resources(
                    4,
                    Some(2),
                    vec![1, 2, 3],
                    [ResourceSnapshot::new(
                        ResourceId(8),
                        ResourceKind::Buffer,
                        "buffer",
                        vec![4, 5, 6],
                    )],
                )
                .expect("test capture should freeze"),
            )
            .expect("test capture should publish");
        let server = LocalIpcServer::start(controller.clone(), LocalIpcConfig::default())
            .expect("server should bind loopback");
        (server, controller)
    }

    fn hello(
        stream: &mut TcpStream,
        endpoint: &Endpoint,
        capabilities: Capabilities,
    ) -> ServerMessage {
        let message = ClientMessage::Hello {
            protocol_version: SUPPORTED_PROTOCOL_VERSION,
            capabilities,
            max_message_bytes: u32::MAX,
            auth_token: *endpoint.auth_token(),
        };
        write_message(stream, &message, crate::protocol::DEFAULT_MAX_MESSAGE_BYTES)
            .expect("hello should write");
        read_message(stream, crate::protocol::DEFAULT_MAX_MESSAGE_BYTES)
            .expect("hello response should read")
    }

    #[test]
    fn authenticated_client_negotiates_and_reads_a_frozen_snapshot() {
        let (mut server, _controller) = connected_server();
        let mut stream =
            TcpStream::connect(server.endpoint().address()).expect("client should connect");
        let negotiated = hello(&mut stream, server.endpoint(), Capabilities::ALL);
        assert!(
            matches!(negotiated, ServerMessage::Hello { capabilities, .. } if capabilities == Capabilities::ALL)
        );
        write_message(
            &mut stream,
            &ClientMessage::Request(Request::Snapshot),
            crate::protocol::DEFAULT_MAX_MESSAGE_BYTES,
        )
        .expect("request should write");
        let response: ServerMessage =
            read_message(&mut stream, crate::protocol::DEFAULT_MAX_MESSAGE_BYTES)
                .expect("snapshot should read");
        assert!(
            matches!(response, ServerMessage::Response(Response::Capture(Some(capture))) if capture.capture_id() == 4)
        );
        server.shutdown().expect("server should shut down");
    }

    #[test]
    fn wrong_cookie_is_rejected_before_any_request_is_served() {
        let (mut server, _controller) = connected_server();
        let mut stream =
            TcpStream::connect(server.endpoint().address()).expect("client should connect");
        let message = ClientMessage::Hello {
            protocol_version: SUPPORTED_PROTOCOL_VERSION,
            capabilities: Capabilities::ALL,
            max_message_bytes: u32::MAX,
            auth_token: [0; 32],
        };
        write_message(
            &mut stream,
            &message,
            crate::protocol::DEFAULT_MAX_MESSAGE_BYTES,
        )
        .expect("hello should write");
        let response: ServerMessage =
            read_message(&mut stream, crate::protocol::DEFAULT_MAX_MESSAGE_BYTES)
                .expect("error should read");
        assert!(matches!(
            response,
            ServerMessage::Error {
                code: ErrorCode::AuthenticationFailed,
                ..
            }
        ));
        server.shutdown().expect("server should shut down");
    }

    #[test]
    fn capability_negotiation_blocks_unrequested_readback() {
        let (mut server, _controller) = connected_server();
        let mut stream =
            TcpStream::connect(server.endpoint().address()).expect("client should connect");
        hello(
            &mut stream,
            server.endpoint(),
            Capabilities::from_capability(Capability::Snapshot),
        );
        write_message(
            &mut stream,
            &ClientMessage::Request(Request::ReadResource {
                id: ResourceId(8),
                offset: 0,
                length: 1,
            }),
            crate::protocol::DEFAULT_MAX_MESSAGE_BYTES,
        )
        .expect("request should write");
        let response: ServerMessage =
            read_message(&mut stream, crate::protocol::DEFAULT_MAX_MESSAGE_BYTES)
                .expect("error should read");
        assert!(matches!(
            response,
            ServerMessage::Error {
                code: ErrorCode::CapabilityUnavailable,
                ..
            }
        ));
        server.shutdown().expect("server should shut down");
    }

    #[test]
    fn resource_readback_is_bounded_and_returns_only_the_requested_range() {
        let (mut server, _controller) = connected_server();
        let mut stream =
            TcpStream::connect(server.endpoint().address()).expect("client should connect");
        hello(
            &mut stream,
            server.endpoint(),
            Capabilities::from_capability(Capability::ResourceReadback),
        );
        write_message(
            &mut stream,
            &ClientMessage::Request(Request::ReadResource {
                id: ResourceId(8),
                offset: 1,
                length: 2,
            }),
            crate::protocol::DEFAULT_MAX_MESSAGE_BYTES,
        )
        .expect("request should write");
        let response: ServerMessage =
            read_message(&mut stream, crate::protocol::DEFAULT_MAX_MESSAGE_BYTES)
                .expect("readback should read");
        assert!(
            matches!(response, ServerMessage::Response(Response::Resource(readback)) if readback.bytes == vec![5, 6])
        );
        server.shutdown().expect("server should shut down");
    }

    #[test]
    fn loopback_endpoint_does_not_accept_a_public_bind_address() {
        let config = LocalIpcConfig {
            bind_address: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 0,
            max_message_bytes: 1024,
            max_connections: 1,
            connection_timeout: DEFAULT_CONNECTION_TIMEOUT,
            capabilities: Capabilities::ALL,
        };
        let server = LocalIpcServer::start(Arc::new(CaptureController::new()), config)
            .expect("loopback bind should work");
        assert_eq!(
            server.endpoint().address().ip(),
            IpAddr::V4(Ipv4Addr::LOCALHOST)
        );
    }

    #[test]
    fn capture_arm_and_stop_are_explicit_control_requests() {
        let controller = Arc::new(CaptureController::new());
        let mut server = LocalIpcServer::start(controller.clone(), LocalIpcConfig::default())
            .expect("server should bind loopback");
        let mut stream =
            TcpStream::connect(server.endpoint().address()).expect("client should connect");
        hello(
            &mut stream,
            server.endpoint(),
            Capabilities::from_capability(Capability::CaptureArm)
                .union(Capabilities::from_capability(Capability::CaptureStop)),
        );
        write_message(
            &mut stream,
            &ClientMessage::Request(Request::ArmCapture),
            crate::protocol::DEFAULT_MAX_MESSAGE_BYTES,
        )
        .expect("arm should write");
        assert!(matches!(
            read_message::<ServerMessage>(&mut stream, crate::protocol::DEFAULT_MAX_MESSAGE_BYTES)
                .expect("arm response should read"),
            ServerMessage::Response(Response::Accepted)
        ));
        assert_eq!(controller.state(), crate::capture::CaptureState::Armed);
        write_message(
            &mut stream,
            &ClientMessage::Request(Request::StopCapture),
            crate::protocol::DEFAULT_MAX_MESSAGE_BYTES,
        )
        .expect("stop should write");
        assert!(matches!(
            read_message::<ServerMessage>(&mut stream, crate::protocol::DEFAULT_MAX_MESSAGE_BYTES)
                .expect("stop response should read"),
            ServerMessage::Response(Response::Accepted)
        ));
        assert_eq!(
            controller.state(),
            crate::capture::CaptureState::StopRequested
        );
        server.shutdown().expect("server should shut down");
    }

    #[test]
    fn public_bind_addresses_are_rejected_before_socket_creation() {
        let config = LocalIpcConfig {
            bind_address: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            ..LocalIpcConfig::default()
        };
        assert!(matches!(
            LocalIpcServer::start(Arc::new(CaptureController::new()), config),
            Err(LocalIpcError::NonLoopbackAddress)
        ));
    }

    #[test]
    fn response_encoding_still_enforces_the_client_limit() {
        let capture = FrozenCapture::with_resources(
            1,
            None,
            vec![1, 2, 3],
            [ResourceSnapshot::new(
                ResourceId(8),
                ResourceKind::Buffer,
                "buffer",
                vec![4, 5, 6],
            )],
        )
        .expect("capture should be valid");
        let response = ServerMessage::Response(Response::Capture(Some(capture)));
        let frame = encode_message(&response, crate::protocol::DEFAULT_MAX_MESSAGE_BYTES)
            .expect("normal response should fit");
        let _: ServerMessage = decode_message(&frame, crate::protocol::DEFAULT_MAX_MESSAGE_BYTES)
            .expect("normal response should decode");
        assert!(encode_message(&response, 8).is_err());
    }
}
