#[cfg(target_family = "wasm")]
pub use web_time::Instant;
#[cfg(not(target_family = "wasm"))]
pub use std::time::Instant;

pub use std::time::Duration;
