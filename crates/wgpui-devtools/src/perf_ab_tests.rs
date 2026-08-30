//! Reusable A/B measurement records; workload construction remains frontend-owned.
use std::time::Duration;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sample {
    pub frames: u64,
    pub elapsed: Duration,
}
impl Sample {
    pub fn per_frame(self) -> Option<Duration> {
        (self.frames != 0).then(|| self.elapsed / self.frames as u32)
    }
}
