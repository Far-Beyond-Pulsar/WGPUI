//! Versioned, bounded messages shared by file and local IPC consumers.

use crate::capture::{FrozenCapture, ResourceId, ResourceReadback};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

pub const SUPPORTED_PROTOCOL_VERSION: u16 = 1;
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
const MAX_FRAME_PREFIX_BYTES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Capability {
    CaptureArm = 0,
    CaptureStop = 1,
    Snapshot = 2,
    ResourceReadback = 3,
}

impl Capability {
    const fn bit(self) -> u64 {
        1u64 << (self as u8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Capabilities(u64);

impl Capabilities {
    pub const NONE: Self = Self(0);
    pub const ALL: Self = Self(
        Capability::CaptureArm.bit()
            | Capability::CaptureStop.bit()
            | Capability::Snapshot.bit()
            | Capability::ResourceReadback.bit(),
    );

    pub const fn from_capability(capability: Capability) -> Self {
        Self(capability.bit())
    }
    pub const fn contains(self, capability: Capability) -> bool {
        self.0 & capability.bit() != 0
    }
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    pub const fn bits(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ClientMessage {
    Hello {
        protocol_version: u16,
        capabilities: Capabilities,
        max_message_bytes: u32,
        auth_token: [u8; 32],
    },
    Request(Request),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    ArmCapture,
    StopCapture,
    Snapshot,
    ReadResource {
        id: ResourceId,
        offset: u64,
        length: u32,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ServerMessage {
    Hello {
        protocol_version: u16,
        capabilities: Capabilities,
        max_message_bytes: u32,
    },
    Response(Response),
    Error {
        code: ErrorCode,
        message: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    Accepted,
    Capture(Option<FrozenCapture>),
    Resource(ResourceReadback),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    AuthenticationFailed,
    UnsupportedVersion,
    CapabilityUnavailable,
    InvalidRequest,
    CaptureUnavailable,
    ResourceUnavailable,
    MessageTooLarge,
    Backpressure,
    Internal,
}

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("message length {length} exceeds the {maximum}-byte limit")]
    MessageTooLarge { length: usize, maximum: usize },
    #[error("message length prefix is invalid")]
    InvalidLength,
    #[error("message encoding failed: {0}")]
    Encoding(String),
    #[error("message decoding failed: {0}")]
    Decoding(String),
}

pub fn encode_message<T: Serialize>(
    message: &T,
    maximum_bytes: usize,
) -> Result<Vec<u8>, ProtocolError> {
    let body =
        serde_json::to_vec(message).map_err(|error| ProtocolError::Encoding(error.to_string()))?;
    if body.len() > maximum_bytes {
        return Err(ProtocolError::MessageTooLarge {
            length: body.len(),
            maximum: maximum_bytes,
        });
    }
    let length = u32::try_from(body.len()).map_err(|_| ProtocolError::MessageTooLarge {
        length: body.len(),
        maximum: maximum_bytes,
    })?;
    let mut frame = Vec::with_capacity(MAX_FRAME_PREFIX_BYTES + body.len());
    frame.extend_from_slice(&length.to_le_bytes());
    frame.extend_from_slice(&body);
    Ok(frame)
}

pub fn decode_message<T: for<'de> Deserialize<'de>>(
    frame: &[u8],
    maximum_bytes: usize,
) -> Result<T, ProtocolError> {
    if frame.len() < MAX_FRAME_PREFIX_BYTES {
        return Err(ProtocolError::InvalidLength);
    }
    let length_bytes: [u8; 4] = frame[..MAX_FRAME_PREFIX_BYTES]
        .try_into()
        .map_err(|_| ProtocolError::InvalidLength)?;
    let length = usize::try_from(u32::from_le_bytes(length_bytes))
        .map_err(|_| ProtocolError::InvalidLength)?;
    if length > maximum_bytes {
        return Err(ProtocolError::MessageTooLarge {
            length,
            maximum: maximum_bytes,
        });
    }
    if frame.len() != MAX_FRAME_PREFIX_BYTES + length {
        return Err(ProtocolError::InvalidLength);
    }
    serde_json::from_slice(&frame[MAX_FRAME_PREFIX_BYTES..])
        .map_err(|error| ProtocolError::Decoding(error.to_string()))
}

pub fn write_message<T: Serialize>(
    writer: &mut impl Write,
    message: &T,
    maximum_bytes: usize,
) -> Result<(), ProtocolError> {
    let frame = encode_message(message, maximum_bytes)?;
    writer.write_all(&frame)?;
    writer.flush()?;
    Ok(())
}

pub fn read_message<T: for<'de> Deserialize<'de>>(
    reader: &mut impl Read,
    maximum_bytes: usize,
) -> Result<T, ProtocolError> {
    let mut length_bytes = [0; MAX_FRAME_PREFIX_BYTES];
    reader.read_exact(&mut length_bytes)?;
    let length = usize::try_from(u32::from_le_bytes(length_bytes))
        .map_err(|_| ProtocolError::InvalidLength)?;
    if length > maximum_bytes {
        return Err(ProtocolError::MessageTooLarge {
            length,
            maximum: maximum_bytes,
        });
    }
    let mut frame = Vec::with_capacity(MAX_FRAME_PREFIX_BYTES + length);
    frame.extend_from_slice(&length_bytes);
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    frame.extend_from_slice(&body);
    decode_message(&frame, maximum_bytes)
}

#[derive(Debug)]
pub struct BoundedMessageQueue<T> {
    sender: std::sync::mpsc::SyncSender<T>,
    receiver: std::sync::mpsc::Receiver<T>,
}

impl<T> BoundedMessageQueue<T> {
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel(capacity);
        Self { sender, receiver }
    }

    pub fn try_push(&self, message: T) -> Result<(), T> {
        self.sender.try_send(message).map_err(|error| match error {
            std::sync::mpsc::TrySendError::Full(message)
            | std::sync::mpsc::TrySendError::Disconnected(message) => message,
        })
    }
    pub fn pop(&self) -> Option<T> {
        self.receiver.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_delimited_messages_round_trip() {
        let message = ClientMessage::Request(Request::ReadResource {
            id: ResourceId(9),
            offset: 2,
            length: 4,
        });
        let frame = encode_message(&message, 1024).expect("message should encode");
        let decoded: ClientMessage = decode_message(&frame, 1024).expect("message should decode");
        assert!(matches!(
            decoded,
            ClientMessage::Request(Request::ReadResource {
                id: ResourceId(9),
                offset: 2,
                length: 4
            })
        ));
    }

    #[test]
    fn malformed_and_oversized_frames_are_rejected_before_allocation() {
        assert!(matches!(
            decode_message::<ClientMessage>(&[1, 0], 1024),
            Err(ProtocolError::InvalidLength)
        ));
        let oversized = [0xff, 0xff, 0xff, 0x7f];
        assert!(matches!(
            decode_message::<ClientMessage>(&oversized, 1024),
            Err(ProtocolError::MessageTooLarge { .. })
        ));
        let message = ClientMessage::Request(Request::Snapshot);
        assert!(matches!(
            encode_message(&message, 1),
            Err(ProtocolError::MessageTooLarge { .. })
        ));
    }

    #[test]
    fn bounded_queue_reports_backpressure_without_dropping_the_message() {
        let queue = BoundedMessageQueue::new(1);
        queue.try_push(1).expect("first message fits");
        assert_eq!(queue.try_push(2), Err(2));
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), None);
    }
}
