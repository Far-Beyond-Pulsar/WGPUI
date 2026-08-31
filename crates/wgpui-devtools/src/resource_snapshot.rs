//! Versioned, bounded resource data for an external inspector.
//!
//! This module is deliberately a data boundary. Buffer input is an owned or
//! borrowed byte slice supplied by a framework adapter; there is no pointer,
//! address, or arbitrary-memory API here. GPU adapters must perform readback
//! and capability checks before handing bytes to this module.

use std::fmt;

use wgpui_core::indirect::{DrawIndirectArgs, DrawSlot};
use wgpui_core::patch::primitive::PrimitiveKind;
use wgpui_core::scene::layer::LayerTable;
use wgpui_core::scene::tile::{TileCoord, TileResidency};

pub const SNAPSHOT_SCHEMA_VERSION: u16 = 1;
pub const SNAPSHOT_MAGIC: [u8; 8] = *b"WGPUIRS1";
pub const SNAPSHOT_HEADER_BYTES: usize = 52;
const SNAPSHOT_SECTION_HEADER_BYTES: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotLimits {
    pub max_payload_bytes: usize,
    pub max_buffer_bytes: usize,
    pub max_hex_bytes: usize,
    pub max_records: usize,
}

impl Default for SnapshotLimits {
    fn default() -> Self {
        Self {
            max_payload_bytes: 1024 * 1024,
            max_buffer_bytes: 64 * 1024,
            max_hex_bytes: 16 * 1024,
            max_records: 16 * 1024,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RedactionPolicy {
    ranges: Vec<ByteRange>,
}

impl RedactionPolicy {
    pub fn new(mut ranges: Vec<ByteRange>) -> Self {
        ranges.sort_unstable_by_key(|range| (range.offset, range.length));
        Self { ranges }
    }

    pub fn ranges(&self) -> &[ByteRange] {
        &self.ranges
    }

    fn redact(&self, offset: usize, bytes: &mut [u8]) -> u64 {
        let end = offset.saturating_add(bytes.len());
        let mut redacted = 0u64;
        for range in &self.ranges {
            let range_end = range.offset.saturating_add(range.length);
            let start = offset.max(range.offset);
            let stop = end.min(range_end);
            if start < stop {
                let local_start = start - offset;
                let local_end = stop - offset;
                bytes[local_start..local_end].fill(0);
                redacted = redacted.saturating_add((local_end - local_start) as u64);
            }
        }
        redacted
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ByteRange {
    pub offset: usize,
    pub length: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TruncationMetadata {
    pub truncated: bool,
    pub omitted_records: u64,
    pub omitted_bytes: u64,
    pub redacted_bytes: u64,
}

impl TruncationMetadata {
    fn add_records(&mut self, records: usize) {
        self.truncated |= records != 0;
        self.omitted_records = self.omitted_records.saturating_add(records as u64);
    }

    fn add_bytes(&mut self, bytes: usize) {
        self.truncated |= bytes != 0;
        self.omitted_bytes = self.omitted_bytes.saturating_add(bytes as u64);
    }

    fn merge(&mut self, other: Self) {
        self.truncated |= other.truncated;
        self.omitted_records = self.omitted_records.saturating_add(other.omitted_records);
        self.omitted_bytes = self.omitted_bytes.saturating_add(other.omitted_bytes);
        self.redacted_bytes = self.redacted_bytes.saturating_add(other.redacted_bytes);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SnapshotError {
    OutOfRange {
        offset: usize,
        length: usize,
        available: usize,
    },
    InvalidAlignment {
        offset: usize,
        length: usize,
        element_size: usize,
    },
    InvalidFormat(&'static str),
    UnsupportedVersion(u16),
    PayloadTooLarge {
        requested: usize,
        limit: usize,
    },
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange {
                offset,
                length,
                available,
            } => write!(
                formatter,
                "buffer view [{offset}..{}) exceeds {available} available bytes",
                offset.saturating_add(*length)
            ),
            Self::InvalidAlignment {
                offset,
                length,
                element_size,
            } => write!(
                formatter,
                "buffer view offset {offset} and length {length} are not aligned to {element_size} bytes"
            ),
            Self::InvalidFormat(reason) => write!(formatter, "invalid resource snapshot: {reason}"),
            Self::UnsupportedVersion(version) => {
                write!(
                    formatter,
                    "unsupported resource snapshot schema version {version}"
                )
            }
            Self::PayloadTooLarge { requested, limit } => {
                write!(
                    formatter,
                    "snapshot payload {requested} exceeds {limit} bytes"
                )
            }
        }
    }
}

impl std::error::Error for SnapshotError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferElementType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
}

impl BufferElementType {
    pub const fn size(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::U8 => 1,
            Self::I8 => 2,
            Self::U16 => 3,
            Self::I16 => 4,
            Self::U32 => 5,
            Self::I32 => 6,
            Self::U64 => 7,
            Self::I64 => 8,
            Self::F32 => 9,
            Self::F64 => 10,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, SnapshotError> {
        match tag {
            1 => Ok(Self::U8),
            2 => Ok(Self::I8),
            3 => Ok(Self::U16),
            4 => Ok(Self::I16),
            5 => Ok(Self::U32),
            6 => Ok(Self::I32),
            7 => Ok(Self::U64),
            8 => Ok(Self::I64),
            9 => Ok(Self::F32),
            10 => Ok(Self::F64),
            _ => Err(SnapshotError::InvalidFormat(
                "unknown typed buffer element type",
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TypedValue {
    Unsigned(u64),
    Signed(i64),
    Float32(f32),
    Float64(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypedBufferView {
    pub element_type: BufferElementType,
    pub values: Vec<TypedValue>,
}

impl TypedBufferView {
    fn decode(bytes: &[u8], element_type: BufferElementType) -> Self {
        let values = bytes
            .chunks_exact(element_type.size())
            .map(|chunk| match element_type {
                BufferElementType::U8 => TypedValue::Unsigned(u64::from(chunk[0])),
                BufferElementType::I8 => {
                    TypedValue::Signed(i64::from(i8::from_le_bytes([chunk[0]])))
                }
                BufferElementType::U16 => {
                    TypedValue::Unsigned(u64::from(u16::from_le_bytes([chunk[0], chunk[1]])))
                }
                BufferElementType::I16 => {
                    TypedValue::Signed(i64::from(i16::from_le_bytes([chunk[0], chunk[1]])))
                }
                BufferElementType::U32 => TypedValue::Unsigned(u64::from(u32::from_le_bytes(
                    chunk.try_into().unwrap_or([0; 4]),
                ))),
                BufferElementType::I32 => TypedValue::Signed(i64::from(i32::from_le_bytes(
                    chunk.try_into().unwrap_or([0; 4]),
                ))),
                BufferElementType::U64 => {
                    TypedValue::Unsigned(u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])))
                }
                BufferElementType::I64 => {
                    TypedValue::Signed(i64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])))
                }
                BufferElementType::F32 => {
                    TypedValue::Float32(f32::from_le_bytes(chunk.try_into().unwrap_or([0; 4])))
                }
                BufferElementType::F64 => {
                    TypedValue::Float64(f64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])))
                }
            })
            .collect();
        Self {
            element_type,
            values,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BufferViewSnapshot {
    pub resource_id: u64,
    pub requested: ByteRange,
    pub total_bytes: u64,
    pub bytes: Vec<u8>,
    pub hex: String,
    pub typed: Option<TypedBufferView>,
    pub truncation: TruncationMetadata,
}

impl BufferViewSnapshot {
    pub fn from_bytes(
        resource_id: u64,
        source: &[u8],
        requested: ByteRange,
        element_type: Option<BufferElementType>,
        limits: SnapshotLimits,
        redaction: &RedactionPolicy,
    ) -> Result<Self, SnapshotError> {
        let end =
            requested
                .offset
                .checked_add(requested.length)
                .ok_or(SnapshotError::OutOfRange {
                    offset: requested.offset,
                    length: requested.length,
                    available: source.len(),
                })?;
        if end > source.len() {
            return Err(SnapshotError::OutOfRange {
                offset: requested.offset,
                length: requested.length,
                available: source.len(),
            });
        }
        if let Some(element_type) = element_type
            && (!requested.offset.is_multiple_of(element_type.size())
                || !requested.length.is_multiple_of(element_type.size()))
        {
            return Err(SnapshotError::InvalidAlignment {
                offset: requested.offset,
                length: requested.length,
                element_size: element_type.size(),
            });
        }
        let mut captured_length = requested.length.min(limits.max_buffer_bytes);
        if let Some(element_type) = element_type {
            captured_length -= captured_length % element_type.size();
        }
        let mut truncation = TruncationMetadata::default();
        truncation.add_bytes(requested.length.saturating_sub(captured_length));
        let mut bytes = source[requested.offset..requested.offset + captured_length].to_vec();
        truncation.redacted_bytes = redaction.redact(requested.offset, &mut bytes);
        let hex_length = bytes.len().min(limits.max_hex_bytes);
        truncation.add_bytes(bytes.len().saturating_sub(hex_length));
        let hex = encode_hex(&bytes[..hex_length]);
        let typed = element_type.map(|element_type| {
            let usable_length = bytes.len() - bytes.len() % element_type.size();
            TypedBufferView::decode(&bytes[..usable_length], element_type)
        });
        Ok(Self {
            resource_id,
            requested,
            total_bytes: source.len() as u64,
            bytes,
            hex,
            typed,
            truncation,
        })
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        result.push(char::from(DIGITS[(byte >> 4) as usize]));
        result.push(char::from(DIGITS[(byte & 0x0f) as usize]));
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileOccupancyRecord {
    pub coord: TileCoord,
    pub resident: bool,
    pub visible: bool,
    pub generation: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileOccupancySnapshot {
    pub records: Vec<TileOccupancyRecord>,
    pub resident_count: u64,
    pub visible_count: u64,
    pub truncation: TruncationMetadata,
}

impl TileOccupancySnapshot {
    pub fn from_residency(
        residency: &TileResidency,
        visible: &[TileCoord],
        limits: SnapshotLimits,
    ) -> Self {
        let mut coords = residency.resident();
        for coord in visible {
            if !coords.contains(coord) {
                coords.push(*coord);
            }
        }
        coords.sort_unstable();
        let visible_set = visible;
        let mut truncation = TruncationMetadata::default();
        let records = coords
            .into_iter()
            .take(limits.max_records)
            .map(|coord| TileOccupancyRecord {
                coord,
                resident: residency.contains(coord),
                visible: visible_set.contains(&coord),
                generation: residency
                    .state(coord)
                    .map(|state| state.last_touch)
                    .unwrap_or(0),
            })
            .collect::<Vec<_>>();
        let total = residency.len().saturating_add(
            visible
                .iter()
                .filter(|coord| !residency.contains(**coord))
                .count(),
        );
        truncation.add_records(total.saturating_sub(records.len()));
        Self {
            records,
            resident_count: residency.len() as u64,
            visible_count: visible.len() as u64,
            truncation,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabAllocationRecord {
    pub owner_id: u64,
    pub generation: u64,
    pub primitive_kind: u8,
    pub base: u32,
    pub capacity: u32,
    pub count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlabMapSnapshot {
    pub records: Vec<SlabAllocationRecord>,
    pub truncation: TruncationMetadata,
}

impl SlabMapSnapshot {
    pub fn from_layer_table(table: &LayerTable, limits: SnapshotLimits) -> Self {
        let mut records = Vec::new();
        for owner_id in table.ids() {
            if let Some(layer) = table.get(owner_id) {
                for primitive_kind in PrimitiveKind::ALL {
                    let range = layer.slab(primitive_kind);
                    if range.capacity != 0 {
                        records.push(SlabAllocationRecord {
                            owner_id: owner_id.as_raw(),
                            generation: layer.generation(),
                            primitive_kind: primitive_kind.index() as u8,
                            base: range.base,
                            capacity: range.capacity,
                            count: range.count,
                        });
                    }
                }
            }
        }
        let mut truncation = TruncationMetadata::default();
        let total = records.len();
        records.truncate(limits.max_records);
        truncation.add_records(total.saturating_sub(records.len()));
        Self {
            records,
            truncation,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasPageRecord {
    pub page_id: u32,
    pub kind: u8,
    pub width: u32,
    pub height: u32,
    pub live_tiles: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasPlacementRecord {
    pub tile_id: u32,
    pub page_id: u32,
    pub kind: u8,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtlasPackingSnapshot {
    pub pages: Vec<AtlasPageRecord>,
    pub placements: Vec<AtlasPlacementRecord>,
    pub truncation: TruncationMetadata,
}

impl AtlasPackingSnapshot {
    pub fn new(
        pages: Vec<AtlasPageRecord>,
        placements: Vec<AtlasPlacementRecord>,
        limits: SnapshotLimits,
    ) -> Self {
        let mut truncation = TruncationMetadata::default();
        let page_count = pages.len();
        let placement_count = placements.len();
        let pages: Vec<_> = pages.into_iter().take(limits.max_records).collect();
        let remaining = limits
            .max_records
            .saturating_sub(page_count.min(limits.max_records));
        let placements: Vec<_> = placements.into_iter().take(remaining).collect();
        truncation.add_records(
            page_count
                .saturating_sub(pages.len())
                .saturating_add(placement_count.saturating_sub(placements.len())),
        );
        Self {
            pages,
            placements,
            truncation,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectDrawRecord {
    pub slot_id: u64,
    pub primitive_kind: u8,
    pub base: u32,
    pub reserved: u32,
    pub args: DrawIndirectArgs,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndirectDrawSnapshot {
    pub records: Vec<IndirectDrawRecord>,
    pub truncation: TruncationMetadata,
}

impl IndirectDrawSnapshot {
    pub fn from_slots(
        slots: &[DrawSlot],
        args: &[DrawIndirectArgs],
        limits: SnapshotLimits,
    ) -> Self {
        let count = slots.len().min(args.len());
        let records = slots
            .iter()
            .zip(args.iter())
            .take(limits.max_records)
            .map(|(slot, args)| IndirectDrawRecord {
                slot_id: slot.layer.as_raw(),
                primitive_kind: slot.kind.index() as u8,
                base: slot.base,
                reserved: slot.count,
                args: *args,
            })
            .collect::<Vec<_>>();
        let mut truncation = TruncationMetadata::default();
        truncation.add_records(count.saturating_sub(records.len()));
        truncation.add_records(slots.len().saturating_sub(count));
        Self {
            records,
            truncation,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResourceSnapshot {
    pub frame_id: u64,
    pub buffers: Vec<BufferViewSnapshot>,
    pub tiles: Option<TileOccupancySnapshot>,
    pub slabs: Option<SlabMapSnapshot>,
    pub atlas: Option<AtlasPackingSnapshot>,
    pub indirect: Option<IndirectDrawSnapshot>,
    pub truncation: TruncationMetadata,
    limits: SnapshotLimits,
}

impl ResourceSnapshot {
    pub fn new(frame_id: u64, limits: SnapshotLimits) -> Self {
        Self {
            frame_id,
            limits,
            ..Self::default()
        }
    }

    pub fn add_buffer(&mut self, buffer: BufferViewSnapshot) {
        if self.buffers.len() >= self.limits.max_records {
            self.truncation.add_records(1);
            return;
        }
        self.truncation.merge(buffer.truncation);
        self.buffers.push(buffer);
    }

    pub fn set_tiles(&mut self, tiles: TileOccupancySnapshot) {
        self.truncation.merge(tiles.truncation);
        self.tiles = Some(tiles);
    }

    pub fn set_slabs(&mut self, slabs: SlabMapSnapshot) {
        self.truncation.merge(slabs.truncation);
        self.slabs = Some(slabs);
    }

    pub fn set_atlas(&mut self, atlas: AtlasPackingSnapshot) {
        self.truncation.merge(atlas.truncation);
        self.atlas = Some(atlas);
    }

    pub fn set_indirect(&mut self, indirect: IndirectDrawSnapshot) {
        self.truncation.merge(indirect.truncation);
        self.indirect = Some(indirect);
    }

    pub fn encode(&self) -> Result<Vec<u8>, SnapshotError> {
        let mut sections = Vec::new();
        if !self.buffers.is_empty() {
            sections.push((1u8, encode_buffers(&self.buffers)?));
        }
        if let Some(tiles) = &self.tiles {
            sections.push((2u8, encode_tiles(tiles)?));
        }
        if let Some(slabs) = &self.slabs {
            sections.push((3u8, encode_slabs(slabs)?));
        }
        if let Some(atlas) = &self.atlas {
            sections.push((4u8, encode_atlas(atlas)?));
        }
        if let Some(indirect) = &self.indirect {
            sections.push((5u8, encode_indirect(indirect)?));
        }
        let body_length = sections
            .iter()
            .try_fold(0usize, |total, (_, section)| {
                total
                    .checked_add(SNAPSHOT_SECTION_HEADER_BYTES)
                    .and_then(|total| total.checked_add(section.len()))
            })
            .ok_or(SnapshotError::PayloadTooLarge {
                requested: usize::MAX,
                limit: self.limits.max_payload_bytes,
            })?;
        if body_length > u32::MAX as usize {
            return Err(SnapshotError::PayloadTooLarge {
                requested: body_length,
                limit: u32::MAX as usize,
            });
        }
        let total_length = SNAPSHOT_HEADER_BYTES.checked_add(body_length).ok_or(
            SnapshotError::PayloadTooLarge {
                requested: usize::MAX,
                limit: self.limits.max_payload_bytes,
            },
        )?;
        if total_length > self.limits.max_payload_bytes {
            return Err(SnapshotError::PayloadTooLarge {
                requested: total_length,
                limit: self.limits.max_payload_bytes,
            });
        }
        let mut output = Vec::with_capacity(total_length);
        output.extend_from_slice(&SNAPSHOT_MAGIC);
        put_u16(&mut output, SNAPSHOT_SCHEMA_VERSION);
        put_u16(&mut output, u16::from(self.truncation.truncated));
        put_u64(&mut output, self.frame_id);
        put_u32(&mut output, sections.len() as u32);
        put_u32(&mut output, body_length as u32);
        put_u64(&mut output, self.truncation.omitted_records);
        put_u64(&mut output, self.truncation.omitted_bytes);
        put_u64(&mut output, self.truncation.redacted_bytes);
        for (kind, section) in sections {
            output.push(kind);
            output.push(0);
            put_u32(&mut output, section_record_count(kind, &section));
            put_u32(&mut output, section.len() as u32);
            output.extend_from_slice(&section);
        }
        Ok(output)
    }

    pub fn decode_header(bytes: &[u8]) -> Result<SnapshotHeader, SnapshotError> {
        if bytes.len() < SNAPSHOT_HEADER_BYTES || bytes.get(..8) != Some(&SNAPSHOT_MAGIC) {
            return Err(SnapshotError::InvalidFormat(
                "bad magic or truncated header",
            ));
        }
        let version = read_u16(bytes, 8)?;
        if version != SNAPSHOT_SCHEMA_VERSION {
            return Err(SnapshotError::UnsupportedVersion(version));
        }
        let section_count = read_u32(bytes, 20)? as usize;
        let body_length = read_u32(bytes, 24)? as usize;
        let expected_length = SNAPSHOT_HEADER_BYTES
            .checked_add(body_length)
            .ok_or(SnapshotError::InvalidFormat("body length overflow"))?;
        if expected_length != bytes.len() {
            return Err(SnapshotError::InvalidFormat("body exceeds input"));
        }
        let mut cursor = SNAPSHOT_HEADER_BYTES;
        for _ in 0..section_count {
            let section_header_end = cursor.checked_add(SNAPSHOT_SECTION_HEADER_BYTES).ok_or(
                SnapshotError::InvalidFormat("section header length overflow"),
            )?;
            let section_header = bytes
                .get(cursor..section_header_end)
                .ok_or(SnapshotError::InvalidFormat("truncated section header"))?;
            if section_header[1] != 0 || !matches!(section_header[0], 1..=5) {
                return Err(SnapshotError::InvalidFormat(
                    "unknown resource snapshot section",
                ));
            }
            let section_length = u32::from_le_bytes([
                section_header[6],
                section_header[7],
                section_header[8],
                section_header[9],
            ]) as usize;
            cursor = cursor
                .checked_add(SNAPSHOT_SECTION_HEADER_BYTES)
                .and_then(|cursor| cursor.checked_add(section_length))
                .ok_or(SnapshotError::InvalidFormat("section length overflow"))?;
            if cursor > bytes.len() {
                return Err(SnapshotError::InvalidFormat("section exceeds body"));
            }
        }
        if cursor != bytes.len() {
            return Err(SnapshotError::InvalidFormat(
                "section count does not cover body",
            ));
        }
        Ok(SnapshotHeader {
            frame_id: read_u64(bytes, 12)?,
            section_count,
            body_length,
            truncated: bytes[10] & 1 != 0,
            omitted_records: read_u64(bytes, 28)?,
            omitted_bytes: read_u64(bytes, 36)?,
            redacted_bytes: read_u64(bytes, 44)?,
        })
    }

    pub fn decode(bytes: &[u8], limits: SnapshotLimits) -> Result<Self, SnapshotError> {
        if bytes.len() > limits.max_payload_bytes {
            return Err(SnapshotError::PayloadTooLarge {
                requested: bytes.len(),
                limit: limits.max_payload_bytes,
            });
        }
        let header = Self::decode_header(bytes)?;
        let mut snapshot = Self {
            frame_id: header.frame_id,
            buffers: Vec::new(),
            tiles: None,
            slabs: None,
            atlas: None,
            indirect: None,
            truncation: TruncationMetadata {
                truncated: header.truncated,
                omitted_records: header.omitted_records,
                omitted_bytes: header.omitted_bytes,
                redacted_bytes: header.redacted_bytes,
            },
            limits,
        };
        let mut cursor = SNAPSHOT_HEADER_BYTES;
        let mut seen = [false; 5];
        for _ in 0..header.section_count {
            let section_header = take_bytes(bytes, &mut cursor, SNAPSHOT_SECTION_HEADER_BYTES)?;
            let kind = section_header[0];
            let section_count = u32::from_le_bytes([
                section_header[2],
                section_header[3],
                section_header[4],
                section_header[5],
            ]);
            let section_length = u32::from_le_bytes([
                section_header[6],
                section_header[7],
                section_header[8],
                section_header[9],
            ]) as usize;
            let section = take_bytes(bytes, &mut cursor, section_length)?;
            let index = usize::from(kind - 1);
            if seen[index] {
                return Err(SnapshotError::InvalidFormat(
                    "duplicate resource snapshot section",
                ));
            }
            seen[index] = true;
            let parsed_count = match kind {
                1 => decode_buffers(section, &mut snapshot)?,
                2 => decode_tiles(section, &mut snapshot)?,
                3 => decode_slabs(section, &mut snapshot)?,
                4 => decode_atlas(section, &mut snapshot)?,
                5 => decode_indirect(section, &mut snapshot)?,
                _ => {
                    return Err(SnapshotError::InvalidFormat(
                        "unknown resource snapshot section",
                    ));
                }
            };
            if parsed_count != section_count {
                return Err(SnapshotError::InvalidFormat(
                    "section record count mismatch",
                ));
            }
        }
        Ok(snapshot)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotHeader {
    pub frame_id: u64,
    pub section_count: usize,
    pub body_length: usize,
    pub truncated: bool,
    pub omitted_records: u64,
    pub omitted_bytes: u64,
    pub redacted_bytes: u64,
}

fn section_record_count(kind: u8, bytes: &[u8]) -> u32 {
    match kind {
        1 | 2 | 3 | 5 => read_u32(bytes, 0).unwrap_or(0),
        4 => {
            let page_count = read_u32(bytes, 0).unwrap_or(0);
            let page_bytes = (page_count as usize).checked_mul(21);
            let placement_offset = page_bytes.and_then(|length| 4usize.checked_add(length));
            let placement_count = placement_offset
                .and_then(|offset| read_u32(bytes, offset).ok())
                .unwrap_or(0);
            page_count.saturating_add(placement_count)
        }
        _ => 0,
    }
}

fn take_bytes<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    length: usize,
) -> Result<&'a [u8], SnapshotError> {
    let end = cursor
        .checked_add(length)
        .ok_or(SnapshotError::InvalidFormat("payload offset overflow"))?;
    let result = bytes
        .get(*cursor..end)
        .ok_or(SnapshotError::InvalidFormat("payload is truncated"))?;
    *cursor = end;
    Ok(result)
}

fn read_cursor_u8(bytes: &[u8], cursor: &mut usize) -> Result<u8, SnapshotError> {
    take_bytes(bytes, cursor, 1).map(|value| value[0])
}

fn read_cursor_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, SnapshotError> {
    let value = take_bytes(bytes, cursor, 4)?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_cursor_i32(bytes: &[u8], cursor: &mut usize) -> Result<i32, SnapshotError> {
    Ok(read_cursor_u32(bytes, cursor)? as i32)
}

fn read_cursor_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, SnapshotError> {
    let value = take_bytes(bytes, cursor, 8)?;
    Ok(u64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}

fn require_record_limit(count: u32, limits: SnapshotLimits) -> Result<usize, SnapshotError> {
    let count = count as usize;
    if count > limits.max_records {
        return Err(SnapshotError::PayloadTooLarge {
            requested: count,
            limit: limits.max_records,
        });
    }
    Ok(count)
}

fn decode_buffers(bytes: &[u8], snapshot: &mut ResourceSnapshot) -> Result<u32, SnapshotError> {
    let mut cursor = 0;
    let count = read_cursor_u32(bytes, &mut cursor)?;
    let count_usize = require_record_limit(count, snapshot.limits)?;
    let mut buffers = Vec::with_capacity(count_usize);
    for _ in 0..count_usize {
        let resource_id = read_cursor_u64(bytes, &mut cursor)?;
        let offset = usize::try_from(read_cursor_u64(bytes, &mut cursor)?)
            .map_err(|_| SnapshotError::InvalidFormat("buffer offset does not fit"))?;
        let length = usize::try_from(read_cursor_u64(bytes, &mut cursor)?)
            .map_err(|_| SnapshotError::InvalidFormat("buffer length does not fit"))?;
        let total_bytes = read_cursor_u64(bytes, &mut cursor)?;
        let byte_count = read_cursor_u32(bytes, &mut cursor)? as usize;
        if byte_count > snapshot.limits.max_buffer_bytes {
            return Err(SnapshotError::PayloadTooLarge {
                requested: byte_count,
                limit: snapshot.limits.max_buffer_bytes,
            });
        }
        let data = take_bytes(bytes, &mut cursor, byte_count)?.to_vec();
        let tag = read_cursor_u8(bytes, &mut cursor)?;
        let typed = if tag == 0 {
            None
        } else {
            let element_type = BufferElementType::from_tag(tag)?;
            if !data.len().is_multiple_of(element_type.size()) {
                return Err(SnapshotError::InvalidFormat(
                    "typed buffer payload is misaligned",
                ));
            }
            Some(TypedBufferView::decode(&data, element_type))
        };
        let hex_length = data.len().min(snapshot.limits.max_hex_bytes);
        buffers.push(BufferViewSnapshot {
            resource_id,
            requested: ByteRange { offset, length },
            total_bytes,
            hex: encode_hex(&data[..hex_length]),
            bytes: data,
            typed,
            truncation: TruncationMetadata::default(),
        });
    }
    if cursor != bytes.len() {
        return Err(SnapshotError::InvalidFormat(
            "buffer section has trailing bytes",
        ));
    }
    snapshot.buffers = buffers;
    Ok(count)
}

fn decode_tiles(bytes: &[u8], snapshot: &mut ResourceSnapshot) -> Result<u32, SnapshotError> {
    let mut cursor = 0;
    let count = read_cursor_u32(bytes, &mut cursor)?;
    let count_usize = require_record_limit(count, snapshot.limits)?;
    let mut records = Vec::with_capacity(count_usize);
    for _ in 0..count_usize {
        let coord = TileCoord::new(
            read_cursor_i32(bytes, &mut cursor)?,
            read_cursor_i32(bytes, &mut cursor)?,
        );
        let resident = read_cursor_u8(bytes, &mut cursor)?;
        let visible = read_cursor_u8(bytes, &mut cursor)?;
        if resident > 1 || visible > 1 {
            return Err(SnapshotError::InvalidFormat("invalid tile occupancy flag"));
        }
        records.push(TileOccupancyRecord {
            coord,
            resident: resident != 0,
            visible: visible != 0,
            generation: read_cursor_u64(bytes, &mut cursor)?,
        });
    }
    if cursor != bytes.len() {
        return Err(SnapshotError::InvalidFormat(
            "tile section has trailing bytes",
        ));
    }
    snapshot.tiles = Some(TileOccupancySnapshot {
        records,
        resident_count: 0,
        visible_count: 0,
        truncation: TruncationMetadata::default(),
    });
    Ok(count)
}

fn decode_slabs(bytes: &[u8], snapshot: &mut ResourceSnapshot) -> Result<u32, SnapshotError> {
    let mut cursor = 0;
    let count = read_cursor_u32(bytes, &mut cursor)?;
    let count_usize = require_record_limit(count, snapshot.limits)?;
    let mut records = Vec::with_capacity(count_usize);
    for _ in 0..count_usize {
        records.push(SlabAllocationRecord {
            owner_id: read_cursor_u64(bytes, &mut cursor)?,
            generation: read_cursor_u64(bytes, &mut cursor)?,
            primitive_kind: read_cursor_u8(bytes, &mut cursor)?,
            base: read_cursor_u32(bytes, &mut cursor)?,
            capacity: read_cursor_u32(bytes, &mut cursor)?,
            count: read_cursor_u32(bytes, &mut cursor)?,
        });
    }
    if cursor != bytes.len() {
        return Err(SnapshotError::InvalidFormat(
            "slab section has trailing bytes",
        ));
    }
    snapshot.slabs = Some(SlabMapSnapshot {
        records,
        truncation: TruncationMetadata::default(),
    });
    Ok(count)
}

fn decode_atlas(bytes: &[u8], snapshot: &mut ResourceSnapshot) -> Result<u32, SnapshotError> {
    let mut cursor = 0;
    let page_count = read_cursor_u32(bytes, &mut cursor)?;
    let page_count_usize = require_record_limit(page_count, snapshot.limits)?;
    let mut pages = Vec::with_capacity(page_count_usize);
    for _ in 0..page_count_usize {
        pages.push(AtlasPageRecord {
            page_id: read_cursor_u32(bytes, &mut cursor)?,
            kind: read_cursor_u8(bytes, &mut cursor)?,
            width: read_cursor_u32(bytes, &mut cursor)?,
            height: read_cursor_u32(bytes, &mut cursor)?,
            live_tiles: read_cursor_u32(bytes, &mut cursor)?,
        });
    }
    let placement_count = read_cursor_u32(bytes, &mut cursor)?;
    let total_count = page_count
        .checked_add(placement_count)
        .ok_or(SnapshotError::InvalidFormat("atlas record count overflow"))?;
    if total_count as usize > snapshot.limits.max_records {
        return Err(SnapshotError::PayloadTooLarge {
            requested: total_count as usize,
            limit: snapshot.limits.max_records,
        });
    }
    let placement_count_usize = placement_count as usize;
    let mut placements = Vec::with_capacity(placement_count_usize);
    for _ in 0..placement_count_usize {
        placements.push(AtlasPlacementRecord {
            tile_id: read_cursor_u32(bytes, &mut cursor)?,
            page_id: read_cursor_u32(bytes, &mut cursor)?,
            kind: read_cursor_u8(bytes, &mut cursor)?,
            x: read_cursor_u32(bytes, &mut cursor)?,
            y: read_cursor_u32(bytes, &mut cursor)?,
            width: read_cursor_u32(bytes, &mut cursor)?,
            height: read_cursor_u32(bytes, &mut cursor)?,
        });
    }
    if cursor != bytes.len() {
        return Err(SnapshotError::InvalidFormat(
            "atlas section has trailing bytes",
        ));
    }
    snapshot.atlas = Some(AtlasPackingSnapshot {
        pages,
        placements,
        truncation: TruncationMetadata::default(),
    });
    Ok(total_count)
}

fn decode_indirect(bytes: &[u8], snapshot: &mut ResourceSnapshot) -> Result<u32, SnapshotError> {
    let mut cursor = 0;
    let count = read_cursor_u32(bytes, &mut cursor)?;
    let count_usize = require_record_limit(count, snapshot.limits)?;
    let mut records = Vec::with_capacity(count_usize);
    for _ in 0..count_usize {
        records.push(IndirectDrawRecord {
            slot_id: read_cursor_u64(bytes, &mut cursor)?,
            primitive_kind: read_cursor_u8(bytes, &mut cursor)?,
            base: read_cursor_u32(bytes, &mut cursor)?,
            reserved: read_cursor_u32(bytes, &mut cursor)?,
            args: DrawIndirectArgs::from_array([
                read_cursor_u32(bytes, &mut cursor)?,
                read_cursor_u32(bytes, &mut cursor)?,
                read_cursor_u32(bytes, &mut cursor)?,
                read_cursor_u32(bytes, &mut cursor)?,
            ]),
        });
    }
    if cursor != bytes.len() {
        return Err(SnapshotError::InvalidFormat(
            "indirect section has trailing bytes",
        ));
    }
    snapshot.indirect = Some(IndirectDrawSnapshot {
        records,
        truncation: TruncationMetadata::default(),
    });
    Ok(count)
}

fn encode_buffers(buffers: &[BufferViewSnapshot]) -> Result<Vec<u8>, SnapshotError> {
    let mut output = Vec::new();
    if buffers.len() > u32::MAX as usize {
        return Err(SnapshotError::PayloadTooLarge {
            requested: buffers.len(),
            limit: u32::MAX as usize,
        });
    }
    put_u32(&mut output, buffers.len() as u32);
    for buffer in buffers {
        put_u64(&mut output, buffer.resource_id);
        put_u64(&mut output, buffer.requested.offset as u64);
        put_u64(&mut output, buffer.requested.length as u64);
        put_u64(&mut output, buffer.total_bytes);
        if buffer.bytes.len() > u32::MAX as usize {
            return Err(SnapshotError::PayloadTooLarge {
                requested: buffer.bytes.len(),
                limit: u32::MAX as usize,
            });
        }
        put_u32(&mut output, buffer.bytes.len() as u32);
        output.extend_from_slice(&buffer.bytes);
        output.push(
            buffer
                .typed
                .as_ref()
                .map(|typed| typed.element_type.tag())
                .unwrap_or(0),
        );
    }
    Ok(output)
}

fn encode_tiles(snapshot: &TileOccupancySnapshot) -> Result<Vec<u8>, SnapshotError> {
    let mut output = Vec::new();
    put_u32(&mut output, snapshot.records.len() as u32);
    for record in &snapshot.records {
        put_i32(&mut output, record.coord.x);
        put_i32(&mut output, record.coord.y);
        output.push(u8::from(record.resident));
        output.push(u8::from(record.visible));
        put_u64(&mut output, record.generation);
    }
    Ok(output)
}

fn encode_slabs(snapshot: &SlabMapSnapshot) -> Result<Vec<u8>, SnapshotError> {
    let mut output = Vec::new();
    put_u32(&mut output, snapshot.records.len() as u32);
    for record in &snapshot.records {
        put_u64(&mut output, record.owner_id);
        put_u64(&mut output, record.generation);
        output.push(record.primitive_kind);
        put_u32(&mut output, record.base);
        put_u32(&mut output, record.capacity);
        put_u32(&mut output, record.count);
    }
    Ok(output)
}

fn encode_atlas(snapshot: &AtlasPackingSnapshot) -> Result<Vec<u8>, SnapshotError> {
    let mut output = Vec::new();
    put_u32(&mut output, snapshot.pages.len() as u32);
    for page in &snapshot.pages {
        put_u32(&mut output, page.page_id);
        output.push(page.kind);
        put_u32(&mut output, page.width);
        put_u32(&mut output, page.height);
        put_u32(&mut output, page.live_tiles);
    }
    put_u32(&mut output, snapshot.placements.len() as u32);
    for placement in &snapshot.placements {
        put_u32(&mut output, placement.tile_id);
        put_u32(&mut output, placement.page_id);
        output.push(placement.kind);
        put_u32(&mut output, placement.x);
        put_u32(&mut output, placement.y);
        put_u32(&mut output, placement.width);
        put_u32(&mut output, placement.height);
    }
    Ok(output)
}

fn encode_indirect(snapshot: &IndirectDrawSnapshot) -> Result<Vec<u8>, SnapshotError> {
    let mut output = Vec::new();
    put_u32(&mut output, snapshot.records.len() as u32);
    for record in &snapshot.records {
        put_u64(&mut output, record.slot_id);
        output.push(record.primitive_kind);
        put_u32(&mut output, record.base);
        put_u32(&mut output, record.reserved);
        for word in record.args.to_array() {
            put_u32(&mut output, word);
        }
    }
    Ok(output)
}

fn put_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}
fn put_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}
fn put_i32(output: &mut Vec<u8>, value: i32) {
    output.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, SnapshotError> {
    let end = offset
        .checked_add(2)
        .ok_or(SnapshotError::InvalidFormat("integer offset overflow"))?;
    let value = bytes
        .get(offset..end)
        .ok_or(SnapshotError::InvalidFormat("truncated integer"))?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, SnapshotError> {
    let end = offset
        .checked_add(4)
        .ok_or(SnapshotError::InvalidFormat("integer offset overflow"))?;
    let value = bytes
        .get(offset..end)
        .ok_or(SnapshotError::InvalidFormat("truncated integer"))?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, SnapshotError> {
    let end = offset
        .checked_add(8)
        .ok_or(SnapshotError::InvalidFormat("integer offset overflow"))?;
    let value = bytes
        .get(offset..end)
        .ok_or(SnapshotError::InvalidFormat("truncated integer"))?;
    Ok(u64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui_core::scene::layer::{BoundaryId, LayerKey};
    use wgpui_core::scene::slab_range::SlabRange;
    use wgpui_core::scene::tile::TileSpan;

    fn limits() -> SnapshotLimits {
        SnapshotLimits {
            max_payload_bytes: 4096,
            max_buffer_bytes: 8,
            max_hex_bytes: 8,
            max_records: 4,
        }
    }

    #[test]
    fn typed_decoding_is_little_endian_and_safe() {
        let bytes = [1, 0, 2, 0, 0, 0, 0, 0];
        let view = BufferViewSnapshot::from_bytes(
            7,
            &bytes,
            ByteRange {
                offset: 0,
                length: 8,
            },
            Some(BufferElementType::U16),
            limits(),
            &RedactionPolicy::default(),
        )
        .expect("valid typed view");
        assert_eq!(
            view.typed.expect("typed data").values,
            vec![
                TypedValue::Unsigned(1),
                TypedValue::Unsigned(2),
                TypedValue::Unsigned(0),
                TypedValue::Unsigned(0)
            ]
        );
        assert_eq!(view.hex, "0100020000000000");
    }

    #[test]
    fn out_of_range_requests_are_errors_not_panics() {
        let result = BufferViewSnapshot::from_bytes(
            1,
            &[1, 2],
            ByteRange {
                offset: 2,
                length: 1,
            },
            None,
            SnapshotLimits::default(),
            &RedactionPolicy::default(),
        );
        assert!(matches!(result, Err(SnapshotError::OutOfRange { .. })));
    }

    #[test]
    fn limits_and_redaction_are_explicit() {
        let view = BufferViewSnapshot::from_bytes(
            1,
            &(0u8..32).collect::<Vec<_>>(),
            ByteRange {
                offset: 0,
                length: 32,
            },
            None,
            limits(),
            &RedactionPolicy::new(vec![ByteRange {
                offset: 2,
                length: 2,
            }]),
        )
        .expect("bounded view");
        assert_eq!(view.bytes.len(), 8);
        assert_eq!(view.hex, "0001000004050607");
        assert!(view.truncation.truncated);
        assert_eq!(view.truncation.omitted_bytes, 24);
        assert_eq!(view.truncation.redacted_bytes, 2);
    }

    #[test]
    fn snapshot_encoding_has_stable_header_and_bounded_payload() {
        let mut snapshot = ResourceSnapshot::new(42, limits());
        snapshot.set_indirect(IndirectDrawSnapshot::from_slots(&[], &[], limits()));
        let encoded = snapshot.encode().expect("small snapshot");
        assert_eq!(&encoded[..8], &SNAPSHOT_MAGIC);
        let header = ResourceSnapshot::decode_header(&encoded).expect("header");
        assert_eq!(header.frame_id, 42);
        assert_eq!(header.section_count, 1);
        assert!(!header.truncated);
    }

    #[test]
    fn redaction_metadata_is_exported_with_the_frame() {
        let buffer = BufferViewSnapshot::from_bytes(
            3,
            &[1, 2, 3, 4],
            ByteRange {
                offset: 0,
                length: 4,
            },
            None,
            SnapshotLimits::default(),
            &RedactionPolicy::new(vec![ByteRange {
                offset: 1,
                length: 1,
            }]),
        )
        .expect("valid buffer view");
        let mut snapshot = ResourceSnapshot::new(9, SnapshotLimits::default());
        snapshot.add_buffer(buffer);
        let header = ResourceSnapshot::decode_header(&snapshot.encode().expect("valid export"))
            .expect("valid header");
        assert_eq!(header.redacted_bytes, 1);
    }

    #[test]
    fn bounded_wire_sections_decode_without_untrusted_allocations() {
        let buffer = BufferViewSnapshot::from_bytes(
            5,
            &[1, 0, 2, 0],
            ByteRange {
                offset: 0,
                length: 4,
            },
            Some(BufferElementType::U16),
            SnapshotLimits::default(),
            &RedactionPolicy::default(),
        )
        .expect("valid buffer view");
        let mut original = ResourceSnapshot::new(12, SnapshotLimits::default());
        original.add_buffer(buffer);
        original.set_indirect(IndirectDrawSnapshot::from_slots(
            &[],
            &[],
            SnapshotLimits::default(),
        ));
        let encoded = original.encode().expect("valid export");
        let decoded =
            ResourceSnapshot::decode(&encoded, SnapshotLimits::default()).expect("valid decode");
        assert_eq!(decoded.frame_id, original.frame_id);
        assert_eq!(decoded.buffers[0].typed, original.buffers[0].typed);
        assert_eq!(decoded.indirect, original.indirect);
    }

    #[test]
    fn native_records_are_snapshotted_without_exposing_live_storage() {
        let mut residency = TileResidency::new(8);
        residency.mark(TileSpan::single(TileCoord::new(-1, 2)), 3);
        let tiles = TileOccupancySnapshot::from_residency(
            &residency,
            &[TileCoord::new(-1, 2), TileCoord::new(4, 5)],
            SnapshotLimits::default(),
        );
        assert_eq!(tiles.records.len(), 2);
        assert_eq!(tiles.records[0].coord, TileCoord::new(-1, 2));
        assert!(tiles.records[0].resident);
        assert!(tiles.records[1].visible);

        let mut layers = LayerTable::new();
        let layer_id = layers.insert(LayerKey::untiled(BoundaryId::ROOT));
        assert!(layers.set_slab(
            layer_id,
            PrimitiveKind::Quad,
            SlabRange {
                base: 4,
                capacity: 64,
                count: 2
            },
        ));
        let slabs = SlabMapSnapshot::from_layer_table(&layers, SnapshotLimits::default());
        assert_eq!(slabs.records.len(), 1);
        assert_eq!(slabs.records[0].base, 4);
        assert_eq!(
            slabs.records[0].generation,
            layers.get(layer_id).expect("live layer").generation()
        );

        let atlas = AtlasPackingSnapshot::new(
            vec![AtlasPageRecord {
                page_id: 0,
                kind: 1,
                width: 128,
                height: 128,
                live_tiles: 1,
            }],
            vec![AtlasPlacementRecord {
                tile_id: 9,
                page_id: 0,
                kind: 1,
                x: 2,
                y: 3,
                width: 8,
                height: 10,
            }],
            SnapshotLimits::default(),
        );
        assert_eq!(atlas.placements[0].tile_id, 9);

        let slot = DrawSlot {
            layer: layer_id,
            kind: PrimitiveKind::Quad,
            base: 4,
            count: 2,
        };
        let indirect = IndirectDrawSnapshot::from_slots(
            &[slot],
            &[DrawIndirectArgs {
                vertex_count: 4,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            }],
            SnapshotLimits::default(),
        );
        assert_eq!(indirect.records[0].args.instance_count, 1);
    }

    #[test]
    fn record_limits_report_omissions() {
        let limits = SnapshotLimits {
            max_records: 1,
            ..SnapshotLimits::default()
        };
        let atlas = AtlasPackingSnapshot::new(
            vec![
                AtlasPageRecord {
                    page_id: 0,
                    kind: 1,
                    width: 1,
                    height: 1,
                    live_tiles: 0,
                },
                AtlasPageRecord {
                    page_id: 1,
                    kind: 1,
                    width: 1,
                    height: 1,
                    live_tiles: 0,
                },
            ],
            vec![AtlasPlacementRecord {
                tile_id: 1,
                page_id: 0,
                kind: 1,
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            }],
            limits,
        );
        assert_eq!(atlas.pages.len(), 1);
        assert!(atlas.truncation.truncated);
        assert_eq!(atlas.truncation.omitted_records, 2);
    }

    #[test]
    fn malformed_snapshot_header_is_rejected_without_indexing() {
        assert!(ResourceSnapshot::decode_header(&[0; 31]).is_err());
        assert!(ResourceSnapshot::decode_header(b"not-a-snapshot").is_err());
    }

    #[test]
    fn payload_limit_is_enforced_before_export() {
        let mut snapshot = ResourceSnapshot::new(
            1,
            SnapshotLimits {
                max_payload_bytes: SNAPSHOT_HEADER_BYTES,
                ..SnapshotLimits::default()
            },
        );
        snapshot.set_indirect(IndirectDrawSnapshot::from_slots(
            &[],
            &[],
            SnapshotLimits::default(),
        ));
        assert!(matches!(
            snapshot.encode(),
            Err(SnapshotError::PayloadTooLarge { .. })
        ));
    }
}
