#![allow(missing_docs)]
use std::time::{Duration, Instant};
use crate::{Axis, Pixels, Point, TouchPhase, px};

/// Tracks the dominant axis of a touch scroll gesture.
#[derive(Clone, Copy, Debug, Default)]
pub struct OngoingScroll { last_event: Option<Instant>, axis: Option<Axis> }
impl OngoingScroll {
    pub fn filter(&mut self, delta: &mut Point<Pixels>, phase: TouchPhase) {
        if matches!(phase, TouchPhase::Ended) { self.last_event=None; self.axis=None; return; }
        let x=delta.x.abs(); let y=delta.y.abs();
        if x==Pixels::ZERO && y==Pixels::ZERO { return; }
        let now=Instant::now();
        let fresh=matches!(phase, TouchPhase::Started) || self.last_event.map_or(true, |t| now.duration_since(t)>=Duration::from_millis(28));
        if fresh { self.axis=Some(if x<=y {Axis::Vertical} else {Axis::Horizontal}); }
        else if x.max(y)>=px(6.) { if matches!(self.axis, Some(Axis::Vertical)) && x>y && x>=y*1.9 {self.axis=None;} if matches!(self.axis, Some(Axis::Horizontal)) && y>x && y>=x*1.9 {self.axis=None;} }
        self.last_event=Some(now);
        match self.axis { Some(Axis::Vertical)=>delta.x=Pixels::ZERO, Some(Axis::Horizontal)=>delta.y=Pixels::ZERO, None=>{} }
    }
}
