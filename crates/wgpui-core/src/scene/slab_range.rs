//! Byte-offset math, size classes, and per-primitive slot addressing (§5.0),
//! plus the pending-upload instruction the delta-upload contract produces.
//! See docs/gpu-native-architecture.md §3.1, §5.0.
//!
//! # Element units, not bytes
//!
//! Every quantity stored here is counted in *slots*, never raw bytes; byte
//! offsets exist only at the use site, derived by multiplying by a kind's
//! stride. The legacy backend's `slab.rs` makes the same commitment and
//! records why: a byte-level allocator can express `offset % stride != 0`,
//! which shipped as silent GPU garbage once already. A slot-level one cannot,
//! because the multiply happens after every placement decision.
//!
//! # No `wgpu` here
//!
//! [`UploadRange`] is the *instruction*, not the call. §3.1 puts the live
//! device in `wgpui-wgpu`; Phase 1's job is to produce a headless, inspectable
//! list of exactly which bytes changed, and a later phase's job is to turn
//! each entry into one `write_buffer`. Keeping it as data is also what makes
//! §5.0's gate checkable at all without a GPU.

use crate::patch::RecordKey;
use crate::patch::primitive::PrimitiveKind;
use crate::scene::layer::LayerId;
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// The smallest size class a slab reservation is ever rounded up to, in slots.
///
/// Every reservation and every free block is a power-of-two multiple of this,
/// so small count wobbles neither move nor resize a layer's range, and a byte
/// offset derived from any base is always a multiple of
/// `slot_stride * MIN_CLASS`.
pub const MIN_CLASS: u32 = 64;

/// Largest representable size class. A request whose class would exceed this
/// is rejected rather than wrapped.
pub const MAX_CLASS: u32 = 1 << 31;

/// Round a slot count up to its size class. `0` maps to `0`: a layer with no
/// slots of a kind holds no reservation at all.
///
/// Returns `None` for a count no class can cover, which the allocator reports
/// as an overflow rather than truncating.
pub fn size_class(count: u32) -> Option<u32> {
    if count == 0 {
        return Some(0);
    }
    if count > MAX_CLASS {
        return None;
    }
    Some(count.max(MIN_CLASS).next_power_of_two())
}

/// A stable range of slots inside one primitive kind's arena.
///
/// `count` is how many slots the owner currently occupies; `capacity` is the
/// size-class-rounded reservation containing them (`capacity >= count`, both
/// zero for [`SlabRange::EMPTY`]). `base` is the index of the first slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SlabRange {
    /// Index of the first reserved slot in the kind's arena.
    pub base: u32,
    /// Size-class-rounded reservation, in slots.
    pub capacity: u32,
    /// Slots actually occupied by the owner's current content.
    pub count: u32,
}

impl SlabRange {
    /// The canonical empty range: an owner with zero slots of a kind holds
    /// this rather than a null marker.
    pub const EMPTY: SlabRange = SlabRange {
        base: 0,
        capacity: 0,
        count: 0,
    };

    /// Whether this range holds no content.
    pub const fn is_empty(self) -> bool {
        self.count == 0
    }

    /// One past the last reserved slot.
    pub const fn end(self) -> u32 {
        self.base + self.capacity
    }

    /// Byte offset of the range start for a kind whose slots are
    /// `slot_stride` bytes wide.
    pub const fn byte_offset(self, slot_stride: usize) -> u64 {
        self.base as u64 * slot_stride as u64
    }

    /// The occupied byte span. A full-layer upload covers exactly this, never
    /// the unused capacity tail.
    pub const fn used_byte_range(self, slot_stride: usize) -> Range<u64> {
        let offset = self.byte_offset(slot_stride);
        offset..offset + self.count as u64 * slot_stride as u64
    }

    /// The full reserved byte span, including capacity slack.
    pub const fn reserved_byte_range(self, slot_stride: usize) -> Range<u64> {
        let offset = self.byte_offset(slot_stride);
        offset..offset + self.capacity as u64 * slot_stride as u64
    }

    /// The byte span of `slot_length` slots starting `slot_offset` slots into
    /// this range — §5.0's "a GPU-side address per primitive, not just a
    /// per-layer range."
    ///
    /// Returns `None` when the requested span runs past the occupied count,
    /// which is a bookkeeping bug in the caller rather than a legal query.
    pub fn slot_byte_range(
        self,
        slot_offset: u32,
        slot_length: u32,
        slot_stride: usize,
    ) -> Option<Range<u64>> {
        let last = slot_offset.checked_add(slot_length)?;
        if last > self.count {
            return None;
        }
        let start = (self.base as u64 + slot_offset as u64) * slot_stride as u64;
        Some(start..start + slot_length as u64 * slot_stride as u64)
    }
}

/// One pending upload: the bytes of one kind's arena that changed and must
/// reach the GPU before the next draw.
///
/// This is the headless half of §5.0's contract. `wgpui-wgpu` (§3.5,
/// `render/buffers/upload.rs`) turns each entry into exactly one
/// `write_buffer(offset, size)`; nothing in `wgpui-core` ever does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct UploadRange {
    /// Which kind's arena these bytes belong to.
    pub kind: PrimitiveKind,
    /// Byte offset into that arena.
    pub byte_offset: u64,
    /// Byte length. Always a whole number of slots.
    pub byte_length: u64,
}

/// The kind of change represented by one primitive-slot diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SlotChange {
    /// A new record now occupies the slots.
    Inserted,
    /// An existing record changed without moving its slots.
    Updated,
    /// A record's slots moved because a neighbouring record changed size or
    /// the allocator relocated the layer.
    Reflowed,
    /// A record no longer occupies the slots.
    Removed,
}

/// An owned description of one primitive record's slot change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PrimitiveSlotDiff {
    /// Primitive kind and arena.
    pub kind: PrimitiveKind,
    /// Owning retained layer.
    pub layer: LayerId,
    /// Stable record address.
    pub key: RecordKey,
    /// Why the record's slots changed.
    pub change: SlotChange,
    /// Previous layer-relative slot span, when the record was resident.
    pub old: Option<SlotSpan>,
    /// Current layer-relative slot span, when the record is resident.
    pub new: Option<SlotSpan>,
}

/// A layer-relative span of primitive slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SlotSpan {
    /// First slot relative to the layer's reservation.
    pub start: u32,
    /// Number of slots occupied.
    pub count: u32,
}

impl UploadRange {
    /// One past the last byte this instruction covers.
    pub const fn byte_end(self) -> u64 {
        self.byte_offset + self.byte_length
    }

    /// Whether this instruction covers no bytes. Callers drop these rather
    /// than emit them: §5.0's third case is "a clean layer uploads zero bytes
    /// — not a small range, zero."
    pub const fn is_empty(self) -> bool {
        self.byte_length == 0
    }
}

/// Merge byte-adjacent and overlapping instructions of the same kind, in
/// place.
///
/// This is §5.0's stated mitigation for its own stated risk — "a burst of many
/// small, scattered, non-adjacent primitive updates ... could regress into
/// many tiny `write_buffer` calls, trading bytes-transferred for driver
/// call-count overhead" — and it is the same adjacency rule the legacy
/// renderer already applies to *draws* (`OpenSlabRun`), applied to writes.
///
/// Genuinely scattered updates stay scattered: coalescing never widens an
/// instruction to cover bytes that did not change, because doing so would
/// re-upload clean primitives and quietly turn the O(1) guarantee back into
/// the per-layer upload it replaces.
pub fn coalesce_uploads(uploads: &mut Vec<UploadRange>) {
    uploads.retain(|upload| !upload.is_empty());
    uploads.sort_unstable();

    let mut merged: Vec<UploadRange> = Vec::with_capacity(uploads.len());
    for upload in uploads.iter().copied() {
        match merged.last_mut() {
            Some(previous)
                if previous.kind == upload.kind && upload.byte_offset <= previous.byte_end() =>
            {
                let end = previous.byte_end().max(upload.byte_end());
                previous.byte_length = end - previous.byte_offset;
            }
            _ => merged.push(upload),
        }
    }
    *uploads = merged;
}

/// Total bytes a set of upload instructions moves.
pub fn uploaded_byte_count(uploads: &[UploadRange]) -> u64 {
    uploads.iter().map(|upload| upload.byte_length).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_classes_round_to_powers_of_two_floored_at_min() {
        assert_eq!(size_class(0), Some(0));
        assert_eq!(size_class(1), Some(MIN_CLASS));
        assert_eq!(size_class(MIN_CLASS), Some(MIN_CLASS));
        assert_eq!(size_class(MIN_CLASS + 1), Some(MIN_CLASS * 2));
        assert_eq!(size_class(1000), Some(1024));
    }

    #[test]
    fn size_class_reports_overflow_instead_of_wrapping() {
        assert_eq!(size_class(MAX_CLASS), Some(MAX_CLASS));
        assert_eq!(size_class(MAX_CLASS + 1), None);
        assert_eq!(size_class(u32::MAX), None);
    }

    #[test]
    fn byte_ranges_stay_stride_aligned() {
        let range = SlabRange {
            base: 128,
            capacity: 128,
            count: 70,
        };
        assert_eq!(range.byte_offset(64), 128 * 64);
        assert_eq!(range.used_byte_range(64), 8192..8192 + 70 * 64);
        assert_eq!(range.reserved_byte_range(64), 8192..8192 + 128 * 64);
        assert_eq!(range.end(), 256);
    }

    #[test]
    fn slot_addressing_targets_one_primitive_not_its_layer() {
        let range = SlabRange {
            base: 64,
            capacity: 128,
            count: 100,
        };
        let one_slot = range.slot_byte_range(9, 1, 64);
        assert_eq!(one_slot, Some((64 + 9) * 64..(64 + 10) * 64));
        assert_eq!(
            one_slot.map(|span| span.end - span.start),
            Some(64),
            "a single-slot address must be exactly one stride wide"
        );
    }

    #[test]
    fn slot_addressing_refuses_to_run_past_the_occupied_count() {
        let range = SlabRange {
            base: 0,
            capacity: 128,
            count: 100,
        };
        assert_eq!(range.slot_byte_range(99, 2, 64), None);
        assert_eq!(range.slot_byte_range(u32::MAX, 1, 64), None);
    }

    #[test]
    fn coalescing_merges_adjacent_ranges_of_the_same_kind() {
        let mut uploads = vec![
            UploadRange {
                kind: PrimitiveKind::Quad,
                byte_offset: 64,
                byte_length: 64,
            },
            UploadRange {
                kind: PrimitiveKind::Quad,
                byte_offset: 0,
                byte_length: 64,
            },
        ];
        coalesce_uploads(&mut uploads);
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].byte_offset, 0);
        assert_eq!(uploads[0].byte_length, 128);
    }

    #[test]
    fn coalescing_leaves_scattered_ranges_scattered() {
        let mut uploads = vec![
            UploadRange {
                kind: PrimitiveKind::Quad,
                byte_offset: 0,
                byte_length: 64,
            },
            UploadRange {
                kind: PrimitiveKind::Quad,
                byte_offset: 6400,
                byte_length: 64,
            },
        ];
        coalesce_uploads(&mut uploads);
        assert_eq!(uploads.len(), 2);
        assert_eq!(uploaded_byte_count(&uploads), 128);
    }

    #[test]
    fn coalescing_never_merges_across_kinds() {
        let mut uploads = vec![
            UploadRange {
                kind: PrimitiveKind::Quad,
                byte_offset: 0,
                byte_length: 64,
            },
            UploadRange {
                kind: PrimitiveKind::GlyphRun,
                byte_offset: 64,
                byte_length: 48,
            },
        ];
        coalesce_uploads(&mut uploads);
        assert_eq!(uploads.len(), 2, "two arenas, two buffers, never one write");
    }

    #[test]
    fn coalescing_drops_empty_instructions() {
        let mut uploads = vec![UploadRange {
            kind: PrimitiveKind::Quad,
            byte_offset: 512,
            byte_length: 0,
        }];
        coalesce_uploads(&mut uploads);
        assert!(uploads.is_empty());
    }

    #[test]
    fn coalescing_merges_overlapping_ranges_without_widening_beyond_them() {
        let mut uploads = vec![
            UploadRange {
                kind: PrimitiveKind::Quad,
                byte_offset: 0,
                byte_length: 128,
            },
            UploadRange {
                kind: PrimitiveKind::Quad,
                byte_offset: 64,
                byte_length: 64,
            },
        ];
        coalesce_uploads(&mut uploads);
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].byte_length, 128);
    }
}
