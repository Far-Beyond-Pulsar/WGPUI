//! Immutable frame diagnostics for damage, retained slots, uploads, and input.
//!
//! The live renderer may use these types as a handoff, but a frozen report owns
//! all of its vectors. Inspecting or serializing it therefore cannot borrow or
//! mutate the application, scene, or input dispatcher.

use crate::geometry::Rect;
use crate::patch::RecordKey;
use crate::patch::primitive::PrimitiveKind;
use crate::scene::{LayerId, PrimitiveSlotDiff, SlotChange, SlotSpan, UploadRange};
use crate::window::InputEvent;
use serde::{Deserialize, Serialize};

/// The source of a damaged region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DamageReason {
    /// Primitive content changed.
    Content,
    /// A hover/active state changed.
    Hover,
    /// A viewport or inherited clip changed.
    ClipResize,
    /// A scroll exposed content that was not previously presented.
    ScrollReveal,
    /// An atlas or other external resource changed.
    Resource,
    /// An externally owned surface is producing a new frame.
    ContinuousSurface,
}

/// One owned damage region in layer coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DamageRegion {
    /// Layer that owns the pixels.
    pub layer: LayerId,
    /// Damaged area in that layer's content space.
    pub bounds: Rect,
    /// Why the area must be considered.
    pub reason: DamageReason,
}

/// A canonical, immutable list of damage regions.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DamageMap {
    regions: Vec<DamageRegion>,
}

impl DamageMap {
    /// An empty map.
    pub const fn new() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// Add a region. Empty and non-finite rectangles are ignored because they
    /// cannot contribute pixels and would make serialized reports unstable.
    pub fn add(&mut self, layer: LayerId, bounds: Rect, reason: DamageReason) {
        if bounds.is_empty()
            || ![bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y]
                .into_iter()
                .all(f32::is_finite)
        {
            return;
        }
        self.regions.push(DamageRegion {
            layer,
            bounds,
            reason,
        });
        self.normalize();
    }

    /// The canonical regions, sorted by stable identity and geometry.
    pub fn regions(&self) -> &[DamageRegion] {
        &self.regions
    }

    /// Whether no pixels are damaged.
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Number of regions.
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Return the union of the old and new region sets, without modifying
    /// either input. This is the exact area a presentation comparison must
    /// inspect when a region moved or disappeared.
    pub fn changed_since(&self, previous: &Self) -> Self {
        let mut changed = Self::new();
        for current in &self.regions {
            if !previous.regions.contains(current) {
                changed.add(current.layer, current.bounds, current.reason);
            }
        }
        for old in &previous.regions {
            if !self.regions.contains(old) {
                changed.add(old.layer, old.bounds, old.reason);
            }
        }
        changed
    }

    fn normalize(&mut self) {
        self.regions.sort_by(|left, right| {
            left.layer
                .cmp(&right.layer)
                .then(left.reason.cmp(&right.reason))
                .then(left.bounds.min_x.total_cmp(&right.bounds.min_x))
                .then(left.bounds.min_y.total_cmp(&right.bounds.min_y))
                .then(left.bounds.max_x.total_cmp(&right.bounds.max_x))
                .then(left.bounds.max_y.total_cmp(&right.bounds.max_y))
        });
        self.regions.dedup();
    }
}

/// A single frame's last-presented/current damage comparison.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FrozenDamage {
    /// Frame number supplied by the presentation owner.
    pub frame_id: u64,
    /// The map that was frozen at the prior successful present.
    pub last_presented: DamageMap,
    /// The map collected for this frame.
    pub current: DamageMap,
    /// Regions that differ between the two maps.
    pub changed: DamageMap,
}

/// Mutable collector owned by a frame loop. Its public snapshot is detached.
#[derive(Clone, Debug, Default)]
pub struct DamageTracker {
    last_presented: DamageMap,
    current: DamageMap,
}

impl DamageTracker {
    /// Start a new frame without discarding the last successful presentation.
    pub fn begin_frame(&mut self) {
        self.current = DamageMap::new();
    }

    /// Record a region in the current frame.
    pub fn add(&mut self, layer: LayerId, bounds: Rect, reason: DamageReason) {
        self.current.add(layer, bounds, reason);
    }

    /// Record the union of a hover region's old and new bounds.
    pub fn record_hover(&mut self, layer: LayerId, old: Rect, new: Rect) {
        self.add(layer, old.union(&new), DamageReason::Hover);
    }

    /// Record clip damage caused by a resize or inherited clip transition.
    pub fn record_resize(&mut self, layer: LayerId, old: Rect, new: Rect) {
        self.add(layer, old.union(&new), DamageReason::ClipResize);
    }

    /// Record only the newly exposed strips of a scrolled content rectangle.
    /// A resident-range scroll can omit this call entirely and remain a pure
    /// transform change.
    pub fn record_scroll_reveal(
        &mut self,
        layer: LayerId,
        viewport: Rect,
        old_offset: [f32; 2],
        new_offset: [f32; 2],
    ) {
        let old_view = translate(viewport, old_offset);
        let new_view = translate(viewport, new_offset);
        for exposed in subtract_rect(new_view, old_view) {
            self.add(layer, exposed, DamageReason::ScrollReveal);
        }
    }

    /// Freeze a report without advancing presentation state.
    pub fn freeze(&self, frame_id: u64) -> FrozenDamage {
        FrozenDamage {
            frame_id,
            last_presented: self.last_presented.clone(),
            current: self.current.clone(),
            changed: self.current.changed_since(&self.last_presented),
        }
    }

    /// Commit the current map as the last successfully presented map and
    /// return the detached report that was visible at that boundary.
    pub fn present(&mut self, frame_id: u64) -> FrozenDamage {
        let frozen = self.freeze(frame_id);
        self.last_presented = self.current.clone();
        self.current = DamageMap::new();
        frozen
    }

    /// The current map, for a non-mutating diagnostic adapter.
    pub fn current(&self) -> &DamageMap {
        &self.current
    }

    /// The last successfully presented map, for a non-mutating diagnostic
    /// adapter.
    pub fn last_presented(&self) -> &DamageMap {
        &self.last_presented
    }
}

fn translate(rect: Rect, offset: [f32; 2]) -> Rect {
    Rect {
        min_x: rect.min_x - offset[0],
        min_y: rect.min_y - offset[1],
        max_x: rect.max_x - offset[0],
        max_y: rect.max_y - offset[1],
    }
}

fn subtract_rect(new_rect: Rect, old_rect: Rect) -> Vec<Rect> {
    let overlap = new_rect.intersect(&old_rect);
    if overlap.is_empty() {
        return vec![new_rect];
    }
    let mut result = Vec::with_capacity(4);
    if new_rect.min_y < overlap.min_y {
        result.push(Rect {
            min_x: new_rect.min_x,
            min_y: new_rect.min_y,
            max_x: new_rect.max_x,
            max_y: overlap.min_y,
        });
    }
    if overlap.max_y < new_rect.max_y {
        result.push(Rect {
            min_x: new_rect.min_x,
            min_y: overlap.max_y,
            max_x: new_rect.max_x,
            max_y: new_rect.max_y,
        });
    }
    if new_rect.min_x < overlap.min_x {
        result.push(Rect {
            min_x: new_rect.min_x,
            min_y: overlap.min_y,
            max_x: overlap.min_x,
            max_y: overlap.max_y,
        });
    }
    if overlap.max_x < new_rect.max_x {
        result.push(Rect {
            min_x: overlap.max_x,
            min_y: overlap.min_y,
            max_x: new_rect.max_x,
            max_y: overlap.max_y,
        });
    }
    result
}

/// A deterministic, owned input event associated with one logical frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayInputRecord {
    /// Presentation frame to which the event was delivered.
    pub frame_id: u64,
    /// Stable order within the frame.
    pub sequence: u32,
    /// Logical timestamp supplied by the input adapter; wall-clock time is
    /// intentionally not captured so serialization remains reproducible.
    pub timestamp_ns: u64,
    /// The event payload.
    pub event: InputEvent,
}

/// Owned single-frame input capture.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SingleFrameInput {
    /// Frame number represented by this record.
    pub frame_id: u64,
    /// Events in dispatch order.
    pub events: Vec<ReplayInputRecord>,
}

impl SingleFrameInput {
    /// Replay as cloned events. This operation has no access to live window or
    /// application state; the caller decides whether and how to dispatch them.
    pub fn replay(&self) -> Vec<InputEvent> {
        self.events
            .iter()
            .map(|record| record.event.clone())
            .collect()
    }
}

/// Mutable recorder for one frame's input boundary.
#[derive(Clone, Debug, Default)]
pub struct InputRecorder {
    frame_id: u64,
    next_sequence: u32,
    events: Vec<ReplayInputRecord>,
}

impl InputRecorder {
    /// Begin recording a frame.
    pub fn begin_frame(&mut self, frame_id: u64) {
        self.frame_id = frame_id;
        self.next_sequence = 0;
        self.events.clear();
    }

    /// Record an owned clone of an event.
    pub fn record(&mut self, timestamp_ns: u64, event: &InputEvent) {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.events.push(ReplayInputRecord {
            frame_id: self.frame_id,
            sequence,
            timestamp_ns,
            event: event.clone(),
        });
    }

    /// Freeze the current records without mutating the recorder.
    pub fn freeze(&self) -> SingleFrameInput {
        SingleFrameInput {
            frame_id: self.frame_id,
            events: self.events.clone(),
        }
    }
}

/// A transport-neutral frozen report for one frame.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FrozenFrameReport {
    /// Damage before and after the frame.
    pub damage: FrozenDamage,
    /// Primitive slot changes.
    pub slot_diffs: Vec<PrimitiveSlotDiff>,
    /// Exact scene-arena byte ranges written by the frame.
    pub uploads: Vec<UploadRange>,
    /// Input delivered for this frame.
    pub input: SingleFrameInput,
}

impl FrozenFrameReport {
    /// Build and canonicalize an owned report from backend outputs.
    pub fn new(
        damage: FrozenDamage,
        mut slot_diffs: Vec<PrimitiveSlotDiff>,
        mut uploads: Vec<UploadRange>,
        input: SingleFrameInput,
    ) -> Self {
        slot_diffs.sort_unstable();
        uploads.sort_unstable();
        Self {
            damage,
            slot_diffs,
            uploads,
            input,
        }
    }

    /// A stable JSON representation, useful for file capture and tests.
    pub fn to_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }
}

/// Construct a slot diff without requiring a live scene, for adapters that
/// already have a slot transition.
pub const fn slot_diff(
    kind: PrimitiveKind,
    layer: LayerId,
    key: RecordKey,
    change: SlotChange,
    old: Option<SlotSpan>,
    new: Option<SlotSpan>,
) -> PrimitiveSlotDiff {
    PrimitiveSlotDiff {
        kind,
        layer,
        key,
        change,
        old,
        new,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::Pixels;
    use crate::window::{InputEvent, Modifiers, MouseMoveEvent};

    const LAYER: LayerId = LayerId::from_raw(7);

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
        Rect::from_origin_size([x, y], [width, height])
    }

    #[test]
    fn frozen_damage_keeps_last_presented_separate_from_current() {
        let mut tracker = DamageTracker::default();
        tracker.begin_frame();
        tracker.add(LAYER, rect(0.0, 0.0, 10.0, 10.0), DamageReason::Content);
        let first = tracker.present(1);
        assert!(first.last_presented.is_empty());
        assert_eq!(first.current.len(), 1);

        tracker.begin_frame();
        tracker.add(LAYER, rect(20.0, 0.0, 10.0, 10.0), DamageReason::Content);
        let frozen = tracker.freeze(2);
        assert_eq!(frozen.last_presented.len(), 1);
        assert_eq!(frozen.current.len(), 1);
        assert_eq!(frozen.changed.len(), 2);
        tracker.add(LAYER, rect(40.0, 0.0, 10.0, 10.0), DamageReason::Content);
        assert_eq!(frozen.current.len(), 1, "freezing must detach the report");
    }

    #[test]
    fn scroll_reveal_reports_strips_but_resident_scroll_can_be_empty() {
        let mut tracker = DamageTracker::default();
        tracker.begin_frame();
        tracker.record_scroll_reveal(LAYER, rect(0.0, 0.0, 100.0, 100.0), [0.0, 0.0], [0.0, 20.0]);
        assert_eq!(
            tracker.current().regions()[0].bounds,
            rect(0.0, -20.0, 100.0, 20.0)
        );
        tracker.begin_frame();
        assert!(tracker.current().is_empty());
    }

    #[test]
    fn resize_shadows_text_and_surfaces_keep_distinct_reasons() {
        let mut tracker = DamageTracker::default();
        tracker.record_resize(
            LAYER,
            rect(0.0, 0.0, 10.0, 10.0),
            rect(0.0, 0.0, 20.0, 20.0),
        );
        tracker.record_hover(LAYER, rect(1.0, 2.0, 5.0, 6.0), rect(3.0, 4.0, 5.0, 6.0));
        tracker.add(LAYER, rect(1.0, 2.0, 5.0, 6.0), DamageReason::Content);
        tracker.add(LAYER, rect(-2.0, -2.0, 15.0, 15.0), DamageReason::Resource);
        tracker.add(
            LAYER,
            rect(30.0, 0.0, 10.0, 10.0),
            DamageReason::ContinuousSurface,
        );
        let map = tracker.current();
        assert_eq!(map.regions().len(), 5);
        assert!(
            map.regions()
                .iter()
                .any(|region| region.reason == DamageReason::Content)
        );
        assert!(
            map.regions()
                .iter()
                .any(|region| region.reason == DamageReason::ClipResize)
        );
        assert!(
            map.regions()
                .iter()
                .any(|region| region.reason == DamageReason::Hover)
        );
        assert!(
            map.regions()
                .iter()
                .any(|region| region.reason == DamageReason::Resource)
        );
        assert!(
            map.regions()
                .iter()
                .any(|region| region.reason == DamageReason::ContinuousSurface)
        );
    }

    #[test]
    fn input_freeze_and_replay_do_not_touch_the_original_event() {
        let event = InputEvent::MouseMove(MouseMoveEvent {
            position: [Pixels(3.0), Pixels(4.0)],
            modifiers: Modifiers::shift(),
            buttons: Default::default(),
        });
        let mut recorder = InputRecorder::default();
        recorder.begin_frame(9);
        recorder.record(12, &event);
        let frozen = recorder.freeze();
        assert_eq!(frozen.replay(), vec![event.clone()]);
        recorder.begin_frame(10);
        assert_eq!(frozen.frame_id, 9);
        assert_eq!(frozen.events[0].frame_id, 9);
    }

    #[test]
    fn frozen_report_json_is_deterministic() {
        let mut first_damage_map = DamageMap::new();
        first_damage_map.add(LAYER, rect(20.0, 0.0, 10.0, 10.0), DamageReason::Resource);
        first_damage_map.add(LAYER, rect(0.0, 0.0, 10.0, 10.0), DamageReason::Content);
        let mut second_damage_map = DamageMap::new();
        second_damage_map.add(LAYER, rect(0.0, 0.0, 10.0, 10.0), DamageReason::Content);
        second_damage_map.add(LAYER, rect(20.0, 0.0, 10.0, 10.0), DamageReason::Resource);

        let first_damage = FrozenDamage {
            frame_id: 3,
            last_presented: DamageMap::new(),
            current: first_damage_map,
            changed: DamageMap::new(),
        };
        let second_damage = FrozenDamage {
            frame_id: 3,
            last_presented: DamageMap::new(),
            current: second_damage_map,
            changed: DamageMap::new(),
        };
        let input = SingleFrameInput {
            frame_id: 3,
            events: vec![ReplayInputRecord {
                frame_id: 3,
                sequence: 0,
                timestamp_ns: 12,
                event: InputEvent::MouseMove(MouseMoveEvent {
                    position: [Pixels(3.0), Pixels(4.0)],
                    modifiers: Modifiers::shift(),
                    buttons: Default::default(),
                }),
            }],
        };
        let first = FrozenFrameReport::new(
            first_damage,
            vec![
                slot_diff(
                    PrimitiveKind::GlyphRun,
                    LAYER,
                    RecordKey::from_raw(2),
                    SlotChange::Reflowed,
                    Some(SlotSpan { start: 4, count: 2 }),
                    Some(SlotSpan { start: 6, count: 2 }),
                ),
                slot_diff(
                    PrimitiveKind::Shadow,
                    LAYER,
                    RecordKey::from_raw(1),
                    SlotChange::Updated,
                    Some(SlotSpan { start: 0, count: 1 }),
                    Some(SlotSpan { start: 0, count: 1 }),
                ),
            ],
            vec![
                UploadRange {
                    kind: PrimitiveKind::GlyphRun,
                    byte_offset: 64,
                    byte_length: 32,
                },
                UploadRange {
                    kind: PrimitiveKind::Shadow,
                    byte_offset: 0,
                    byte_length: 64,
                },
            ],
            input.clone(),
        );
        let second = FrozenFrameReport::new(
            second_damage,
            first.slot_diffs.iter().rev().copied().collect(),
            first.uploads.iter().rev().copied().collect(),
            input,
        );
        assert_eq!(
            first.to_json().expect("report serializes"),
            second.to_json().expect("report serializes")
        );
    }
}
