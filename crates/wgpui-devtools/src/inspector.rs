//! Versioned, transport-neutral retained UI snapshots.
//!
//! The native renderer owns the live plan, layout tree, and scene. This module
//! copies inspectable facts at an explicit frame boundary so a consumer never
//! needs to dereference application state or keep the renderer alive. Fields
//! the current core cannot prove remain optional and are not synthesized.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use wgpui_core::boundary::policy::{BoundaryPolicy, Buffering};
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::patch::primitive::PrimitiveKind;
use wgpui_core::reconcile::{FramePlan, InstanceTable, NodeOutcome, RebuildReason};
use wgpui_core::scene::{LayerId, LayerKey, Scene};
use wgpui_layout::taffy_tree::LayoutTree;

/// The current wire-schema version.
pub const INSPECTOR_SNAPSHOT_SCHEMA_VERSION: u32 = 1;
/// Alias used by capture consumers that treat the framed message as a boxed
/// protocol value.
pub const BOXED_SNAPSHOT_VERSION: u32 = INSPECTOR_SNAPSHOT_SCHEMA_VERSION;
const MAX_SNAPSHOT_BYTES: usize = 128 * 1024 * 1024;

/// The stable address of a retained element plus the generation of the record
/// occupying that address.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StableAddress {
    pub address: u64,
    pub generation: u64,
}

impl StableAddress {
    pub const fn new(address: u64, generation: u64) -> Self {
        Self {
            address,
            generation,
        }
    }
}

/// A source location, when a frontend has one to contribute.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
}

/// Human-readable metadata associated with an element.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementMetadata {
    pub label: Option<String>,
    pub type_name: Option<String>,
    pub source: Option<SourceLocation>,
    pub explicit_id: Option<String>,
    pub retained: bool,
    pub outcome: Option<String>,
}

/// A rectangle in logical-pixel coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RectSnapshot {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl RectSnapshot {
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn from_core(rect: wgpui_core::geometry::Rect) -> Self {
        Self::new(rect.min_x, rect.min_y, rect.width(), rect.height())
    }
}

/// Box-model rectangles. `None` means the backend has not supplied that part
/// of the box model, rather than claiming it is zero-sized.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoxModelSnapshot {
    pub margin: Option<RectSnapshot>,
    pub border: Option<RectSnapshot>,
    pub padding: Option<RectSnapshot>,
    pub content: Option<RectSnapshot>,
}

/// Layout and effective visibility data for an element.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LayoutSnapshot {
    pub bounds: Option<RectSnapshot>,
    pub visible_bounds: Option<RectSnapshot>,
    pub effective_clip: Option<RectSnapshot>,
    pub box_model: BoxModelSnapshot,
}

/// Local and accumulated transforms. The current core uses translations, but
/// the affine shape leaves room for richer adapters without changing the wire
/// schema.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransformSnapshot {
    pub local: [f32; 6],
    pub accumulated: [f32; 6],
    pub scroll_translation: [f32; 2],
}

impl Default for TransformSnapshot {
    fn default() -> Self {
        Self {
            local: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            accumulated: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            scroll_translation: [0.0, 0.0],
        }
    }
}

/// Effective clip information, when the shared walk has supplied it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClipSnapshot {
    pub effective: Option<RectSnapshot>,
    pub ancestors: Vec<RectSnapshot>,
}

/// The four invalidation axes plus an optional reason supplied by a caller.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvalidationSnapshot {
    pub bits: u8,
    pub axes: Vec<String>,
    pub reason: Option<String>,
}

impl InvalidationSnapshot {
    pub fn from_core(invalidation: Invalidation) -> Self {
        let mut axes = Vec::new();
        for (axis, name) in [
            (Invalidation::LAYOUT, "layout"),
            (Invalidation::DISPLAY, "display"),
            (Invalidation::HIT, "hit"),
            (Invalidation::TRANSFORM, "transform"),
        ] {
            if invalidation.contains(axis) {
                axes.push(name.to_string());
            }
        }
        Self {
            bits: invalidation.bits(),
            axes,
            reason: None,
        }
    }
}

/// A retained element in pre-order tree form.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ElementSnapshot {
    pub stable: StableAddress,
    pub parent: Option<StableAddress>,
    pub children: Vec<StableAddress>,
    pub metadata: ElementMetadata,
    pub layout: LayoutSnapshot,
    pub transform: TransformSnapshot,
    pub clip: ClipSnapshot,
    pub scroll_root: Option<u64>,
    pub boundary: Option<u64>,
    pub tile: Option<TileOwnershipSnapshot>,
    pub invalidation: InvalidationSnapshot,
    pub computed_style: Option<ComputedStyle>,
    pub paint_records: Vec<u64>,
    pub last_presented: bool,
}

/// A style representation independent of Taffy's internal Rust types.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ComputedStyle {
    pub properties: BTreeMap<String, String>,
    pub raw: Option<String>,
}

impl ComputedStyle {
    pub fn from_raw(raw: impl Into<String>) -> Self {
        Self {
            properties: BTreeMap::new(),
            raw: Some(raw.into()),
        }
    }
}

/// A scroll root's retained geometry and ownership metadata.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ScrollRootSnapshot {
    pub id: u64,
    pub parent: Option<u64>,
    pub element: Option<StableAddress>,
    pub viewport: Option<RectSnapshot>,
    pub content_bounds: Option<RectSnapshot>,
    pub offset: [f32; 2],
    pub previous_offset: Option<[f32; 2]>,
    pub effective_clip: Option<RectSnapshot>,
    pub boundary: Option<u64>,
    pub resident_tiles: Vec<TileOwnershipSnapshot>,
}

/// Serializable subset of a boundary policy.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoundaryPolicySnapshot {
    pub rasterize_above: usize,
    pub evict_after_frames: u32,
    pub resident_tile_budget: usize,
    pub buffering: BufferingSnapshot,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BufferingSnapshot {
    pub kind: String,
    pub tile_size: Option<[f32; 2]>,
    pub retain_radius: u32,
    pub margin: Option<[f32; 2]>,
}

/// One compositing boundary, including the scene layer generation that backs
/// it when that layer is live.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoundarySnapshot {
    pub id: u64,
    pub element: Option<StableAddress>,
    pub layer: Option<u64>,
    pub generation: u64,
    pub retention: Option<String>,
    pub policy: Option<BoundaryPolicySnapshot>,
    pub transform: TransformSnapshot,
    pub invalidation: InvalidationSnapshot,
}

/// A tile coordinate with its owning boundary/layer generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileOwnershipSnapshot {
    pub boundary: u64,
    pub x: i32,
    pub y: i32,
    pub layer: u64,
    pub generation: u64,
}

/// A resident or visible tile and its bounded residency facts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TileSnapshot {
    pub ownership: TileOwnershipSnapshot,
    pub bounds: Option<RectSnapshot>,
    pub resident: bool,
    pub visible: bool,
    pub last_visited_frame: Option<u64>,
    pub last_touch: Option<u64>,
}

/// A paint record and its exact retained byte range. Bytes make a file capture
/// independently useful while addresses keep it useful for diff tooling.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PaintRecord {
    pub layer: u64,
    pub boundary: u64,
    pub tile: Option<[i32; 2]>,
    pub layer_generation: u64,
    pub kind: String,
    pub record_key: u64,
    pub slot_base: u32,
    pub slot_count: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub bytes: Vec<u8>,
}

/// Presentation facts for the captured frame.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LastPresentedState {
    pub frame_number: Option<u64>,
    pub presented: bool,
    pub present_count: Option<u64>,
    pub viewport: Option<RectSnapshot>,
    pub scene_generation: Option<u64>,
}

impl LastPresentedState {
    pub const fn not_presented() -> Self {
        Self {
            frame_number: None,
            presented: false,
            present_count: None,
            viewport: None,
            scene_generation: None,
        }
    }

    pub const fn presented(frame_number: u64) -> Self {
        Self {
            frame_number: Some(frame_number),
            presented: true,
            present_count: None,
            viewport: None,
            scene_generation: None,
        }
    }
}

/// One complete retained inspector frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InspectorSnapshot {
    pub schema_version: u32,
    pub frame_number: u64,
    pub elements: Vec<ElementSnapshot>,
    pub scroll_roots: Vec<ScrollRootSnapshot>,
    pub boundaries: Vec<BoundarySnapshot>,
    pub tiles: Vec<TileSnapshot>,
    pub invalidation: Vec<InvalidationSnapshot>,
    pub paint_records: Vec<PaintRecord>,
    pub last_presented: LastPresentedState,
}

impl Default for InspectorSnapshot {
    fn default() -> Self {
        Self {
            schema_version: INSPECTOR_SNAPSHOT_SCHEMA_VERSION,
            frame_number: 0,
            elements: Vec::new(),
            scroll_roots: Vec::new(),
            boundaries: Vec::new(),
            tiles: Vec::new(),
            invalidation: Vec::new(),
            paint_records: Vec::new(),
            last_presented: LastPresentedState::default(),
        }
    }
}

impl InspectorSnapshot {
    pub fn new(frame_number: u64) -> Self {
        Self {
            schema_version: INSPECTOR_SNAPSHOT_SCHEMA_VERSION,
            frame_number,
            ..Self::default()
        }
    }

    /// Build a snapshot from native retained stages at a safe boundary.
    /// Layout/style failures are represented as absent optional fields so an
    /// inspector can still show the rest of a frozen frame.
    pub fn from_frame(
        frame_number: u64,
        plan: &FramePlan,
        instances: &InstanceTable,
        layout: &LayoutTree,
        scene: &Scene,
    ) -> Self {
        Self::from_frame_with_presented(
            frame_number,
            LastPresentedState::presented(frame_number),
            plan,
            instances,
            layout,
            scene,
        )
    }

    pub fn from_frame_with_presented(
        frame_number: u64,
        last_presented: LastPresentedState,
        plan: &FramePlan,
        instances: &InstanceTable,
        layout: &LayoutTree,
        scene: &Scene,
    ) -> Self {
        let mut snapshot = Self::new(frame_number);
        snapshot.last_presented = last_presented.clone();
        snapshot.elements = element_snapshots(plan, instances, layout, last_presented.presented);
        snapshot.boundaries = boundary_snapshots(plan, scene, instances);
        snapshot.scroll_roots = scroll_root_snapshots(plan, &snapshot.elements);
        snapshot.paint_records = paint_records(scene);
        snapshot.tiles = tile_snapshots(scene);
        snapshot.invalidation = snapshot
            .elements
            .iter()
            .map(|element| element.invalidation.clone())
            .collect();
        snapshot
    }

    /// Serialize as JSON without transport framing.
    pub fn to_json(&self) -> Result<String, SnapshotError> {
        self.validate_version()?;
        serde_json::to_string_pretty(self).map_err(SnapshotError::Json)
    }

    /// Parse JSON and reject versions this crate cannot interpret.
    pub fn from_json(json: &str) -> Result<Self, SnapshotError> {
        let snapshot: Self = serde_json::from_str(json).map_err(SnapshotError::Json)?;
        snapshot.validate_version()?;
        Ok(snapshot)
    }

    /// Encode one length-delimited capture: little-endian byte length followed
    /// by the UTF-8 JSON payload.
    pub fn to_bytes(&self) -> Result<Vec<u8>, SnapshotError> {
        self.validate_version()?;
        let json = serde_json::to_vec(self).map_err(SnapshotError::Json)?;
        if json.len() > MAX_SNAPSHOT_BYTES {
            return Err(SnapshotError::TooLarge);
        }
        let length = u32::try_from(json.len()).map_err(|_| SnapshotError::TooLarge)?;
        let mut bytes = Vec::with_capacity(4 + json.len());
        bytes.extend_from_slice(&length.to_le_bytes());
        bytes.extend_from_slice(&json);
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SnapshotError> {
        let length_bytes = bytes.get(..4).ok_or(SnapshotError::InvalidFrame)?;
        let length = u32::from_le_bytes([
            length_bytes[0],
            length_bytes[1],
            length_bytes[2],
            length_bytes[3],
        ]) as usize;
        if length > MAX_SNAPSHOT_BYTES || bytes.len() != length.saturating_add(4) {
            return Err(SnapshotError::InvalidFrame);
        }
        let payload = bytes.get(4..).ok_or(SnapshotError::InvalidFrame)?;
        let snapshot: Self = serde_json::from_slice(payload).map_err(SnapshotError::Json)?;
        snapshot.validate_version()?;
        Ok(snapshot)
    }

    pub fn write_file(&self, path: impl AsRef<Path>) -> Result<(), SnapshotError> {
        fs::write(path, self.to_bytes()?).map_err(SnapshotError::Io)
    }

    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, SnapshotError> {
        let bytes = fs::read(path).map_err(SnapshotError::Io)?;
        Self::from_bytes(&bytes)
    }

    fn validate_version(&self) -> Result<(), SnapshotError> {
        if self.schema_version == INSPECTOR_SNAPSHOT_SCHEMA_VERSION {
            Ok(())
        } else {
            Err(SnapshotError::UnsupportedVersion {
                found: self.schema_version,
                supported: INSPECTOR_SNAPSHOT_SCHEMA_VERSION,
            })
        }
    }
}

/// An immutable wrapper used after a capture reaches the presentation
/// boundary. The value remains usable after the renderer is dropped.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenCapture {
    snapshot: InspectorSnapshot,
}

impl FrozenCapture {
    pub fn new(snapshot: InspectorSnapshot) -> Result<Self, SnapshotError> {
        snapshot.validate_version()?;
        Ok(Self { snapshot })
    }

    pub fn snapshot(&self) -> &InspectorSnapshot {
        &self.snapshot
    }

    pub fn into_snapshot(self) -> InspectorSnapshot {
        self.snapshot
    }

    pub fn write_file(&self, path: impl AsRef<Path>) -> Result<(), SnapshotError> {
        self.snapshot.write_file(path)
    }

    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, SnapshotError> {
        Self::new(InspectorSnapshot::read_file(path)?)
    }
}

pub type FrozenInspectorSnapshot = FrozenCapture;

/// Errors from snapshot validation and file/transport encoding.
#[derive(Debug)]
pub enum SnapshotError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidFrame,
    TooLarge,
    UnsupportedVersion { found: u32, supported: u32 },
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "snapshot I/O: {error}"),
            Self::Json(error) => write!(formatter, "snapshot JSON: {error}"),
            Self::InvalidFrame => formatter.write_str("invalid length-delimited snapshot"),
            Self::TooLarge => formatter.write_str("snapshot exceeds the 128 MiB limit"),
            Self::UnsupportedVersion { found, supported } => write!(
                formatter,
                "snapshot schema version {found} is unsupported (expected {supported})"
            ),
        }
    }
}

impl std::error::Error for SnapshotError {}

/// The original small inspector record remains source-compatible. New users
/// can additionally retain a full `InspectorSnapshot`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementInfo {
    pub label: String,
    pub source_file: String,
    pub source_line: u32,
    pub depth: u32,
}

#[derive(Debug, Default)]
pub struct Inspector {
    elements: Vec<ElementInfo>,
    snapshot: Option<FrozenCapture>,
}

impl Inspector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn replace_elements(&mut self, elements: Vec<ElementInfo>) {
        self.elements = elements;
    }

    pub fn elements(&self) -> &[ElementInfo] {
        &self.elements
    }

    pub fn replace_snapshot(&mut self, snapshot: InspectorSnapshot) -> Result<(), SnapshotError> {
        self.snapshot = Some(FrozenCapture::new(snapshot)?);
        Ok(())
    }

    pub fn snapshot(&self) -> Option<&InspectorSnapshot> {
        self.snapshot.as_ref().map(FrozenCapture::snapshot)
    }

    pub fn frozen_capture(&self) -> Option<&FrozenCapture> {
        self.snapshot.as_ref()
    }

    pub fn write_snapshot(&self, path: impl AsRef<Path>) -> Result<(), SnapshotError> {
        self.snapshot
            .as_ref()
            .ok_or(SnapshotError::InvalidFrame)?
            .write_file(path)
    }

    pub fn load_snapshot(&mut self, path: impl AsRef<Path>) -> Result<(), SnapshotError> {
        self.snapshot = Some(FrozenCapture::read_file(path)?);
        Ok(())
    }
}

fn element_snapshots(
    plan: &FramePlan,
    instances: &InstanceTable,
    layout: &LayoutTree,
    presented: bool,
) -> Vec<ElementSnapshot> {
    let mut elements: Vec<ElementSnapshot> = Vec::with_capacity(plan.nodes().len());
    let mut ancestors: Vec<StableAddress> = Vec::new();
    for node in plan.nodes() {
        ancestors.truncate(node.depth as usize);
        let retained = node.instance.and_then(|instance| instances.get(instance));
        let stable = StableAddress::new(
            node.address.as_raw(),
            retained.map_or(0, |instance| instance.generation()),
        );
        let parent = ancestors.last().copied();
        let bounds = layout.layout_of(node.layout_node).ok().map(|rectangle| {
            RectSnapshot::new(rectangle.x, rectangle.y, rectangle.width, rectangle.height)
        });
        let computed_style = layout
            .style_of(node.layout_node)
            .ok()
            .map(|style| ComputedStyle::from_raw(format!("{style:?}")));
        let element = ElementSnapshot {
            stable,
            parent,
            metadata: ElementMetadata {
                type_name: retained
                    .and_then(|instance| instance.type_name())
                    .map(str::to_string),
                retained: node.instance.is_some(),
                outcome: Some(outcome_name(node.outcome).to_string()),
                ..ElementMetadata::default()
            },
            layout: LayoutSnapshot {
                bounds,
                ..LayoutSnapshot::default()
            },
            transform: TransformSnapshot {
                scroll_translation: node.scroll_offset,
                ..TransformSnapshot::default()
            },
            scroll_root: node.declared_boundary.map(|boundary| boundary.as_raw()),
            boundary: Some(node.boundary.as_raw()),
            invalidation: InvalidationSnapshot::from_core(node.invalidation),
            computed_style,
            last_presented: presented,
            ..ElementSnapshot::default()
        };
        if let Some(parent) = parent
            && let Some(parent_element) = elements.iter_mut().find(|item| item.stable == parent)
        {
            parent_element.children.push(stable);
        }
        ancestors.push(stable);
        elements.push(element);
    }
    elements
}

fn outcome_name(outcome: NodeOutcome) -> &'static str {
    match outcome {
        NodeOutcome::Reused => "reused",
        NodeOutcome::Rebuilt(reason) => rebuild_reason_name(reason),
        NodeOutcome::Uncached => "uncached",
    }
}

fn rebuild_reason_name(reason: RebuildReason) -> &'static str {
    match reason {
        RebuildReason::NewInstance => "new_instance",
        RebuildReason::TypeMismatch => "type_mismatch",
        RebuildReason::NoDiffKey => "no_diff_key",
        RebuildReason::KeyChanged => "key_changed",
        RebuildReason::ChildrenChanged => "children_changed",
        RebuildReason::AncestorRebuilt => "ancestor_rebuilt",
        RebuildReason::ReconciliationDisabled => "reconciliation_disabled",
    }
}

fn boundary_snapshots(
    plan: &FramePlan,
    scene: &Scene,
    instances: &InstanceTable,
) -> Vec<BoundarySnapshot> {
    let mut boundaries = Vec::new();
    for node in plan.nodes() {
        let Some(boundary) = node.declared_boundary else {
            continue;
        };
        if boundaries
            .iter()
            .any(|entry: &BoundarySnapshot| entry.id == boundary.as_raw())
        {
            continue;
        }
        let layer_id = scene.layers.ids().into_iter().find(|layer| {
            scene
                .layers
                .get(*layer)
                .is_some_and(|record| record.key() == LayerKey::untiled(boundary))
        });
        let layer = layer_id.and_then(|id| scene.layers.get(id));
        let element_generation = node
            .instance
            .and_then(|instance| instances.get(instance))
            .map_or(0, |instance| instance.generation());
        boundaries.push(BoundarySnapshot {
            id: boundary.as_raw(),
            element: Some(StableAddress::new(
                node.address.as_raw(),
                element_generation,
            )),
            layer: layer_id.map(LayerId::as_raw),
            generation: layer.map_or(0, |record| record.generation()),
            retention: node
                .boundary_policy
                .map(|policy| format!("{:?}", policy.retention_for(0))),
            policy: node.boundary_policy.map(policy_snapshot),
            transform: layer
                .map(|record| TransformSnapshot {
                    scroll_translation: record.transform().translation,
                    ..TransformSnapshot::default()
                })
                .unwrap_or_else(|| TransformSnapshot {
                    scroll_translation: node.scroll_offset,
                    ..TransformSnapshot::default()
                }),
            invalidation: layer
                .map(|record| InvalidationSnapshot::from_core(record.invalidation()))
                .unwrap_or_default(),
        });
    }
    boundaries
}

fn scroll_root_snapshots(
    plan: &FramePlan,
    elements: &[ElementSnapshot],
) -> Vec<ScrollRootSnapshot> {
    let mut roots = Vec::new();
    for node in plan.nodes() {
        let Some(id) = node.declared_boundary else {
            continue;
        };
        if roots
            .iter()
            .any(|root: &ScrollRootSnapshot| root.id == id.as_raw())
        {
            continue;
        }
        let element = elements
            .iter()
            .find(|element| element.stable.address == node.address.as_raw())
            .map(|element| element.stable);
        roots.push(ScrollRootSnapshot {
            id: id.as_raw(),
            parent: Some(node.boundary.as_raw()),
            element,
            offset: node.scroll_offset,
            boundary: Some(id.as_raw()),
            ..ScrollRootSnapshot::default()
        });
    }
    roots
}

fn policy_snapshot(policy: BoundaryPolicy) -> BoundaryPolicySnapshot {
    let buffering = match policy.buffering {
        Buffering::None => BufferingSnapshot {
            kind: "none".to_string(),
            ..BufferingSnapshot::default()
        },
        Buffering::Margin(margin) => BufferingSnapshot {
            kind: "margin".to_string(),
            margin: margin.map(|size| [size.width.value(), size.height.value()]),
            ..BufferingSnapshot::default()
        },
        Buffering::Tiled {
            tile_size,
            retain_radius,
        } => BufferingSnapshot {
            kind: "tiled".to_string(),
            tile_size: Some([tile_size.width.value(), tile_size.height.value()]),
            retain_radius,
            ..BufferingSnapshot::default()
        },
    };
    BoundaryPolicySnapshot {
        rasterize_above: policy.rasterize_above,
        evict_after_frames: policy.evict_after_frames,
        resident_tile_budget: policy.resident_tile_budget,
        buffering,
    }
}

fn tile_snapshots(scene: &Scene) -> Vec<TileSnapshot> {
    scene
        .layers
        .ids()
        .into_iter()
        .filter_map(|layer_id| {
            let layer = scene.layers.get(layer_id)?;
            let key = layer.key();
            let tile = key.tile?;
            let ownership = TileOwnershipSnapshot {
                boundary: key.boundary.as_raw(),
                x: tile.x,
                y: tile.y,
                layer: layer_id.as_raw(),
                generation: layer.generation(),
            };
            Some(TileSnapshot {
                ownership,
                resident: true,
                ..TileSnapshot::default()
            })
        })
        .collect()
}

fn paint_records(scene: &Scene) -> Vec<PaintRecord> {
    let mut records = Vec::new();
    for layer_id in scene.layers.ids() {
        let Some(layer) = scene.layers.get(layer_id) else {
            continue;
        };
        let key = layer.key();
        for_each_paint_store(scene, layer_id, |kind, record_key, range, bytes| {
            let byte_offset = range.start;
            let byte_length = range.end.saturating_sub(range.start);
            let stride = kind.slot_stride() as u64;
            records.push(PaintRecord {
                layer: layer_id.as_raw(),
                boundary: key.boundary.as_raw(),
                tile: key.tile.map(|tile| [tile.x, tile.y]),
                layer_generation: layer.generation(),
                kind: primitive_kind_name(kind).to_string(),
                record_key,
                slot_base: u32::try_from(byte_offset / stride).unwrap_or(u32::MAX),
                slot_count: u32::try_from(byte_length / stride).unwrap_or(u32::MAX),
                byte_offset,
                byte_length,
                bytes,
            });
        });
    }
    records
}

fn for_each_paint_store(
    scene: &Scene,
    layer: LayerId,
    mut visit: impl FnMut(PrimitiveKind, u64, std::ops::Range<u64>, Vec<u8>),
) {
    macro_rules! visit_store {
        ($store:expr, $kind:expr) => {
            for key in $store.keys(layer) {
                let Some(range) = $store.record_byte_range(layer, key) else {
                    continue;
                };
                let Some(bytes) = scene_bytes($store.resident_bytes(), &range) else {
                    continue;
                };
                visit($kind, key.as_raw(), range, bytes);
            }
        };
    }
    visit_store!(scene.shadows, PrimitiveKind::Shadow);
    visit_store!(scene.quads, PrimitiveKind::Quad);
    visit_store!(scene.paths, PrimitiveKind::Path);
    visit_store!(scene.underlines, PrimitiveKind::Underline);
    visit_store!(scene.glyph_runs, PrimitiveKind::GlyphRun);
    visit_store!(scene.poly_sprites, PrimitiveKind::PolySprite);
    visit_store!(scene.backdrop_filters, PrimitiveKind::BackdropFilter);
}

fn scene_bytes(bytes: &[u8], range: &std::ops::Range<u64>) -> Option<Vec<u8>> {
    let start = usize::try_from(range.start).ok()?;
    let end = usize::try_from(range.end).ok()?;
    Some(bytes.get(start..end)?.to_vec())
}

fn primitive_kind_name(kind: PrimitiveKind) -> &'static str {
    match kind {
        PrimitiveKind::Shadow => "shadow",
        PrimitiveKind::Quad => "quad",
        PrimitiveKind::Path => "path",
        PrimitiveKind::Underline => "underline",
        PrimitiveKind::GlyphRun => "glyph_run",
        PrimitiveKind::PolySprite => "poly_sprite",
        PrimitiveKind::BackdropFilter => "backdrop_filter",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui_core::patch::apply::{ScenePatch, apply};
    use wgpui_core::patch::primitive::{Primitive, Quad};
    use wgpui_core::reconcile::{AlwaysDirty, Description, Reconciler};
    use wgpui_core::scene::{BoundaryId, LayerKey};
    use wgpui_layout::taffy_tree::definite;

    struct Panel;

    #[test]
    fn a_snapshot_round_trips_through_json_and_length_delimited_bytes() -> Result<(), SnapshotError>
    {
        let mut snapshot = InspectorSnapshot::new(7);
        snapshot.elements.push(ElementSnapshot {
            stable: StableAddress::new(42, 3),
            metadata: ElementMetadata {
                type_name: Some("Panel".to_string()),
                retained: true,
                ..ElementMetadata::default()
            },
            ..ElementSnapshot::default()
        });
        let json = snapshot.to_json()?;
        assert_eq!(InspectorSnapshot::from_json(&json)?, snapshot);
        assert_eq!(
            InspectorSnapshot::from_bytes(&snapshot.to_bytes()?)?,
            snapshot
        );
        Ok(())
    }

    #[test]
    fn unsupported_versions_are_rejected_before_use() {
        let mut snapshot = InspectorSnapshot::new(1);
        snapshot.schema_version += 1;
        assert!(matches!(
            snapshot.to_json(),
            Err(SnapshotError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn malformed_length_prefixes_do_not_parse_partial_data() -> Result<(), SnapshotError> {
        let snapshot = InspectorSnapshot::new(1);
        let mut bytes = snapshot.to_bytes()?;
        bytes[0] = bytes[0].wrapping_add(1);
        assert!(matches!(
            InspectorSnapshot::from_bytes(&bytes),
            Err(SnapshotError::InvalidFrame)
        ));
        assert!(matches!(
            InspectorSnapshot::from_bytes(&[0, 0, 0]),
            Err(SnapshotError::InvalidFrame)
        ));
        Ok(())
    }

    #[test]
    fn frozen_captures_round_trip_through_a_file() -> Result<(), SnapshotError> {
        let snapshot = InspectorSnapshot::new(19);
        let frozen = FrozenCapture::new(snapshot.clone())?;
        let path = std::env::temp_dir().join(format!(
            "wgpui-inspector-{}-{}.capture",
            std::process::id(),
            snapshot.frame_number
        ));
        frozen.write_file(&path)?;
        let loaded = FrozenCapture::read_file(&path)?;
        std::fs::remove_file(&path).map_err(SnapshotError::Io)?;
        assert_eq!(loaded.snapshot(), &snapshot);
        Ok(())
    }

    #[test]
    fn a_frozen_capture_is_independent_of_the_source_value() -> Result<(), SnapshotError> {
        let snapshot = InspectorSnapshot::new(3);
        let frozen = FrozenCapture::new(snapshot.clone())?;
        assert_eq!(frozen.snapshot(), &snapshot);
        assert_eq!(frozen.into_snapshot(), snapshot);
        Ok(())
    }

    #[test]
    fn legacy_element_records_remain_available_alongside_snapshots() {
        let mut inspector = Inspector::new();
        inspector.replace_elements(vec![ElementInfo {
            label: "root".to_string(),
            source_file: "app.rs".to_string(),
            source_line: 4,
            depth: 0,
        }]);
        assert_eq!(inspector.elements()[0].label, "root");
    }

    #[test]
    fn native_capture_contains_retained_tree_style_generation_and_paint_bytes()
    -> Result<(), SnapshotError> {
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let description = Description::new::<Panel>().diff_key(AlwaysDirty);
        let plan = reconciler
            .reconcile(description, &mut layout)
            .map_err(|_| SnapshotError::InvalidFrame)?;
        let root = plan
            .root()
            .map(|node| node.layout_node)
            .ok_or(SnapshotError::InvalidFrame)?;
        layout
            .compute_layout(root, definite(100.0, 80.0))
            .map_err(|_| SnapshotError::InvalidFrame)?;

        let mut scene = Scene::new();
        let layer = scene.layer(LayerKey::untiled(BoundaryId::ROOT));
        let mut patch = ScenePatch::new();
        patch.quads.append(
            layer,
            wgpui_core::patch::RecordKey::from_raw(9),
            0,
            Quad::ZERO,
        );
        apply(&mut scene, &patch).map_err(|_| SnapshotError::InvalidFrame)?;

        let snapshot =
            InspectorSnapshot::from_frame(11, &plan, reconciler.instances(), &layout, &scene);
        assert_eq!(snapshot.elements.len(), 1);
        assert!(snapshot.elements[0].stable.generation > 0);
        assert_eq!(
            snapshot.elements[0].metadata.type_name.as_deref(),
            Some(std::any::type_name::<Panel>())
        );
        assert!(snapshot.elements[0].computed_style.is_some());
        assert_eq!(snapshot.paint_records.len(), 1);
        assert_eq!(snapshot.paint_records[0].kind, "quad");
        assert_eq!(
            snapshot.paint_records[0].byte_length,
            Quad::SLOT_STRIDE as u64
        );
        assert_eq!(snapshot.paint_records[0].bytes.len(), Quad::SLOT_STRIDE);
        Ok(())
    }
}
