//! Opt-in retained tile and damage diagnostics.
//!
//! The collector stores descriptions of work that already happened in the
//! compositor. It never owns layers, primitive bytes, or GPU resources, so
//! enabling it cannot create a second render cache or broaden invalidation.

use crate::geometry::Rect;
use crate::scene::layer::LayerTransform;
use crate::scene::tile::{TileCoord, TileGrid};
use std::collections::{BTreeMap, BTreeSet};

/// Stable identity for a retained scroll root in diagnostic data.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScrollRootId(u64);

impl ScrollRootId {
    /// The root that owns the window plane.
    pub const ROOT: Self = Self(0);

    /// Construct an identity from an application-owned value.
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Return the application-owned value.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// Why a region was considered damaged.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DamageReason {
    /// Primitive or resource content changed.
    Content,
    /// A hover or interaction state changed.
    Hover,
    /// An effective clip or viewport changed.
    Clip,
    /// Scrolling exposed content that was not resident.
    ScrollReveal,
    /// An external resource changed.
    Resource,
}

/// A requested damage region before child-root subtraction.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DamageRegion {
    /// The root whose content owns this region.
    pub root: ScrollRootId,
    /// Region in the root's content/window coordinate space.
    pub content_rect: Rect,
    /// Why the region needs work.
    pub reason: DamageReason,
}

impl DamageRegion {
    /// Create a damage region from one content rectangle.
    pub const fn new(root: ScrollRootId, content_rect: Rect, reason: DamageReason) -> Self {
        Self {
            root,
            content_rect,
            reason,
        }
    }

    /// Combine the old and new hit regions for one hover transition.
    pub fn hover(root: ScrollRootId, old_hit_region: Rect, new_hit_region: Rect) -> Self {
        Self::new(
            root,
            old_hit_region.union(&new_hit_region),
            DamageReason::Hover,
        )
    }
}

/// The exact raster and compositing consequences of one damage request.
#[derive(Clone, Debug, PartialEq)]
pub struct DamagePlan {
    /// The original request.
    pub damage: DamageRegion,
    /// Rectangles the owning root may rasterize after child coverage is
    /// removed. Multiple rectangles are required for a disjoint remainder.
    pub raster_rects: Vec<Rect>,
    /// Tiles touched by those raster rectangles, when the root has a grid.
    pub raster_tiles: Vec<TileCoord>,
    /// Parent compositing still needs to consider the complete requested quad.
    pub compositing_rect: Rect,
    /// Child roots whose coverage removed parent raster work.
    pub subtracted_children: Vec<ScrollRootId>,
}

/// Inputs describing a root's retained geometry and tile state.
#[derive(Clone, Debug, PartialEq)]
pub struct DebugRootInput<'a> {
    /// Root identity.
    pub id: ScrollRootId,
    /// Parent root, if this is nested.
    pub parent: Option<ScrollRootId>,
    /// Root viewport in the capture coordinate space.
    pub viewport: Rect,
    /// Clip contributed directly by this root.
    pub clip: Rect,
    /// Current compositor transform.
    pub transform: LayerTransform,
    /// Tile grid, if this root owns one.
    pub grid: Option<TileGrid>,
    /// Content-space rectangle under the viewport.
    pub content_viewport: Option<Rect>,
    /// Tiles intersecting the visible range, including retention policy.
    pub visible_tiles: &'a [TileCoord],
    /// Tiles currently resident in memory.
    pub resident_tiles: &'a [TileCoord],
    /// Tiles newly exposed since the previous visit.
    pub newly_exposed_tiles: &'a [TileCoord],
}

/// A root's retained geometry and tile state at capture time.
#[derive(Clone, Debug, PartialEq)]
pub struct DebugRoot {
    /// Root identity.
    pub id: ScrollRootId,
    /// Parent root, if this is nested.
    pub parent: Option<ScrollRootId>,
    /// Root viewport in the capture coordinate space.
    pub viewport: Rect,
    /// Clip contributed directly by this root.
    pub clip: Rect,
    /// Clip after intersecting this root and all known ancestor clips.
    pub effective_clip: Rect,
    /// Current compositor transform.
    pub transform: LayerTransform,
    /// Tile grid, if this root owns one.
    pub grid: Option<TileGrid>,
    /// Content-space rectangle under the viewport.
    pub content_viewport: Option<Rect>,
    /// Tiles intersecting the visible range, including retention policy.
    pub visible_tiles: Vec<TileCoord>,
    /// Tiles currently resident in memory.
    pub resident_tiles: Vec<TileCoord>,
    /// Tiles newly exposed since the previous visit.
    pub newly_exposed_tiles: Vec<TileCoord>,
}

/// One tile's screen-space diagnostic state.
#[derive(Clone, Debug, PartialEq)]
pub struct DebugTile {
    /// Owning root.
    pub root: ScrollRootId,
    /// Tile coordinate in the owning content plane.
    pub coord: TileCoord,
    /// Tile bounds before its root transform.
    pub content_bounds: Rect,
    /// Tile bounds after its root transform.
    pub screen_bounds: Rect,
    /// Effective clip at the time of capture.
    pub effective_clip: Rect,
    /// Whether the tile is eligible to draw this frame.
    pub visible: bool,
    /// Whether the tile has retained content.
    pub resident: bool,
    /// Whether this frame exposed the tile.
    pub newly_exposed: bool,
    /// Whether content in this tile belongs to this root rather than an
    /// ancestor. This stays explicit for nested-root visualizers.
    pub owns_content: bool,
}

/// Captured diagnostic state for one compositor snapshot.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DebugSnapshot {
    /// Roots in stable identity order.
    pub roots: Vec<DebugRoot>,
    /// Expanded tile metadata in stable root/coordinate order.
    pub tiles: Vec<DebugTile>,
    /// Damage plans captured in submission order.
    pub damage: Vec<DamagePlan>,
    /// Number of child-covered regions removed from parent raster damage.
    pub parent_damage_subtractions: usize,
}

/// Opt-in collector for retained tile and damage metadata.
///
/// A disabled collector keeps no maps or vectors. Callers can keep one on a
/// compositor or frame driver and turn it on only while inspecting a frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DebugCapture {
    enabled: bool,
    roots: BTreeMap<ScrollRootId, DebugRoot>,
    damage: Vec<DamagePlan>,
    parent_damage_subtractions: usize,
}

impl DebugCapture {
    /// A disabled collector.
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            roots: BTreeMap::new(),
            damage: Vec::new(),
            parent_damage_subtractions: 0,
        }
    }

    /// An enabled collector with no captured state.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::disabled()
        }
    }

    /// Whether capture is enabled.
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable capture and discard the previous frame's metadata.
    pub fn enable(&mut self) {
        self.enabled = true;
        self.clear();
    }

    /// Disable capture and release its metadata.
    pub fn disable(&mut self) {
        self.enabled = false;
        self.clear();
    }

    /// Discard captured metadata while preserving the enabled state.
    pub fn clear(&mut self) {
        self.roots.clear();
        self.damage.clear();
        self.parent_damage_subtractions = 0;
    }

    /// Record one root and its latest tile visit.
    pub fn record_root(&mut self, input: DebugRootInput<'_>) {
        if !self.enabled {
            return;
        }
        let mut visible_tiles = input.visible_tiles.to_vec();
        let mut resident_tiles = input.resident_tiles.to_vec();
        let mut newly_exposed_tiles = input.newly_exposed_tiles.to_vec();
        visible_tiles.sort_unstable();
        resident_tiles.sort_unstable();
        newly_exposed_tiles.sort_unstable();
        self.roots.insert(
            input.id,
            DebugRoot {
                id: input.id,
                parent: input.parent,
                viewport: input.viewport,
                clip: input.clip,
                effective_clip: input.viewport.intersect(&input.clip),
                transform: input.transform,
                grid: input.grid,
                content_viewport: input.content_viewport,
                visible_tiles,
                resident_tiles,
                newly_exposed_tiles,
            },
        );
        self.recompute_effective_clips();
    }

    fn recompute_effective_clips(&mut self) {
        let ids: Vec<ScrollRootId> = self.roots.keys().copied().collect();
        for id in ids {
            let effective_clip = effective_clip_for(id, &self.roots, &mut BTreeSet::new());
            if let Some(root) = self.roots.get_mut(&id) {
                root.effective_clip = effective_clip;
            }
        }
    }

    /// Record a damage request after subtracting the coverage of direct child
    /// roots. `compositing_rect` intentionally remains unsubtracted.
    pub fn record_damage(&mut self, damage: DamageRegion) -> Option<&DamagePlan> {
        if !self.enabled || damage.content_rect.is_empty() {
            return None;
        }
        let children: Vec<(ScrollRootId, Rect)> = self
            .roots
            .values()
            .filter(|root| root.parent == Some(damage.root))
            .map(|root| (root.id, root.effective_clip))
            .collect();
        let mut raster_rects = vec![damage.content_rect];
        let mut subtracted_children = Vec::new();
        for (child, coverage) in children {
            let mut remainder = Vec::new();
            let mut changed = false;
            for rectangle in raster_rects.iter().copied() {
                let pieces = subtract_rect(rectangle, coverage);
                changed |= pieces.len() != 1 || pieces.first() != Some(&rectangle);
                remainder.extend(pieces);
            }
            if changed {
                subtracted_children.push(child);
                raster_rects = remainder;
            }
        }
        self.parent_damage_subtractions += subtracted_children.len();
        let mut raster_tiles = Vec::new();
        if let Some(root) = self.roots.get(&damage.root)
            && let Some(grid) = root.grid
        {
            for rectangle in &raster_rects {
                if let Some(span) = grid.span(*rectangle) {
                    raster_tiles.extend(span.tiles());
                }
            }
            raster_tiles.sort_unstable();
            raster_tiles.dedup();
        }
        self.damage.push(DamagePlan {
            damage,
            raster_rects,
            raster_tiles,
            compositing_rect: damage.content_rect,
            subtracted_children,
        });
        self.damage.last()
    }

    /// Capture a snapshot whose ordering is deterministic and independent of
    /// map iteration order.
    pub fn snapshot(&self) -> DebugSnapshot {
        if !self.enabled {
            return DebugSnapshot::default();
        }
        let roots: Vec<DebugRoot> = self.roots.values().cloned().collect();
        let mut tiles = Vec::new();
        for root in &roots {
            let Some(grid) = root.grid else {
                continue;
            };
            let mut coordinates = root.resident_tiles.clone();
            coordinates.extend(root.visible_tiles.iter().copied());
            coordinates.sort_unstable();
            coordinates.dedup();
            for coord in coordinates {
                let content_bounds = grid.tile_bounds(coord);
                let translation = root.transform.translation;
                let screen_bounds = Rect::from_origin_size(
                    [
                        content_bounds.min_x + translation[0],
                        content_bounds.min_y + translation[1],
                    ],
                    [content_bounds.width(), content_bounds.height()],
                );
                tiles.push(DebugTile {
                    root: root.id,
                    coord,
                    content_bounds,
                    screen_bounds,
                    effective_clip: root.effective_clip,
                    visible: root.visible_tiles.binary_search(&coord).is_ok()
                        && screen_bounds.intersects(&root.effective_clip),
                    resident: root.resident_tiles.binary_search(&coord).is_ok(),
                    newly_exposed: root.newly_exposed_tiles.binary_search(&coord).is_ok(),
                    owns_content: true,
                });
            }
        }
        DebugSnapshot {
            roots,
            tiles,
            damage: self.damage.clone(),
            parent_damage_subtractions: self.parent_damage_subtractions,
        }
    }
}

fn effective_clip_for(
    id: ScrollRootId,
    roots: &BTreeMap<ScrollRootId, DebugRoot>,
    visiting: &mut BTreeSet<ScrollRootId>,
) -> Rect {
    let Some(root) = roots.get(&id) else {
        return Rect::EMPTY;
    };
    let own_clip = root.viewport.intersect(&root.clip);
    let Some(parent) = root.parent else {
        return own_clip;
    };
    if !visiting.insert(id) {
        return own_clip;
    }
    let parent_clip = if roots.contains_key(&parent) {
        effective_clip_for(parent, roots, visiting)
    } else {
        own_clip
    };
    visiting.remove(&id);
    own_clip.intersect(&parent_clip)
}

/// Subtract one axis-aligned rectangle from another, retaining exact disjoint
/// pieces rather than inflating the result to a bounding box.
fn subtract_rect(source: Rect, covered: Rect) -> Vec<Rect> {
    let overlap = source.intersect(&covered);
    if overlap.is_empty() {
        return vec![source];
    }
    let mut result = Vec::with_capacity(4);
    let top = Rect {
        min_x: source.min_x,
        min_y: source.min_y,
        max_x: source.max_x,
        max_y: overlap.min_y,
    };
    let bottom = Rect {
        min_x: source.min_x,
        min_y: overlap.max_y,
        max_x: source.max_x,
        max_y: source.max_y,
    };
    let left = Rect {
        min_x: source.min_x,
        min_y: overlap.min_y,
        max_x: overlap.min_x,
        max_y: overlap.max_y,
    };
    let right = Rect {
        min_x: overlap.max_x,
        min_y: overlap.min_y,
        max_x: source.max_x,
        max_y: overlap.max_y,
    };
    for piece in [top, bottom, left, right] {
        if !piece.is_empty() {
            result.push(piece);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
        Rect::from_origin_size([x, y], [width, height])
    }

    fn grid() -> TileGrid {
        TileGrid::square(100.0).expect("100px is a valid grid")
    }

    #[test]
    fn disabled_capture_has_no_snapshot_state() {
        let mut capture = DebugCapture::disabled();
        capture.record_root(DebugRootInput {
            id: ScrollRootId::ROOT,
            parent: None,
            viewport: rect(0.0, 0.0, 100.0, 100.0),
            clip: rect(0.0, 0.0, 100.0, 100.0),
            transform: LayerTransform::IDENTITY,
            grid: Some(grid()),
            content_viewport: Some(rect(0.0, 0.0, 100.0, 100.0)),
            visible_tiles: &[TileCoord::ORIGIN],
            resident_tiles: &[TileCoord::ORIGIN],
            newly_exposed_tiles: &[],
        });
        assert_eq!(capture.snapshot(), DebugSnapshot::default());
    }

    #[test]
    fn nested_root_capture_reports_transforms_clips_and_ownership() {
        let mut capture = DebugCapture::enabled();
        capture.record_root(DebugRootInput {
            id: ScrollRootId::ROOT,
            parent: None,
            viewport: rect(0.0, 0.0, 300.0, 300.0),
            clip: rect(0.0, 0.0, 300.0, 300.0),
            transform: LayerTransform::translated(-20.0, 0.0),
            grid: Some(grid()),
            content_viewport: Some(rect(20.0, 0.0, 300.0, 300.0)),
            visible_tiles: &[TileCoord::new(0, 0)],
            resident_tiles: &[TileCoord::new(0, 0), TileCoord::new(1, 0)],
            newly_exposed_tiles: &[TileCoord::new(1, 0)],
        });
        capture.record_root(DebugRootInput {
            id: ScrollRootId::from_raw(1),
            parent: Some(ScrollRootId::ROOT),
            viewport: rect(50.0, 50.0, 100.0, 100.0),
            clip: rect(50.0, 50.0, 80.0, 100.0),
            transform: LayerTransform::translated(10.0, -5.0),
            grid: Some(grid()),
            content_viewport: Some(rect(-10.0, 5.0, 100.0, 100.0)),
            visible_tiles: &[TileCoord::new(0, 0)],
            resident_tiles: &[TileCoord::new(0, 0)],
            newly_exposed_tiles: &[],
        });
        let snapshot = capture.snapshot();
        let child = snapshot
            .roots
            .iter()
            .find(|root| root.id == ScrollRootId::from_raw(1))
            .expect("child root is captured");
        assert_eq!(child.effective_clip, rect(50.0, 50.0, 80.0, 100.0));
        let child_tile = snapshot
            .tiles
            .iter()
            .find(|tile| tile.root == ScrollRootId::from_raw(1))
            .expect("child tile is captured");
        assert_eq!(child_tile.screen_bounds, rect(10.0, -5.0, 100.0, 100.0));
        assert!(child_tile.visible);
        assert!(child_tile.owns_content);
    }

    #[test]
    fn damage_subtraction_preserves_disjoint_parent_remainders_and_compositing() {
        let mut capture = DebugCapture::enabled();
        let parent = ScrollRootId::ROOT;
        let child = ScrollRootId::from_raw(1);
        capture.record_root(DebugRootInput {
            id: child,
            parent: Some(parent),
            viewport: rect(40.0, 20.0, 20.0, 60.0),
            clip: rect(40.0, 20.0, 20.0, 60.0),
            transform: LayerTransform::IDENTITY,
            grid: None,
            content_viewport: None,
            visible_tiles: &[],
            resident_tiles: &[],
            newly_exposed_tiles: &[],
        });
        let plan = capture
            .record_damage(DamageRegion {
                root: parent,
                content_rect: rect(0.0, 0.0, 100.0, 100.0),
                reason: DamageReason::Content,
            })
            .expect("enabled capture records damage")
            .clone();
        assert_eq!(plan.compositing_rect, rect(0.0, 0.0, 100.0, 100.0));
        assert_eq!(plan.subtracted_children, vec![child]);
        assert_eq!(plan.raster_rects.len(), 4);
        assert!(
            plan.raster_rects
                .iter()
                .all(|rectangle| !rectangle.intersects(&rect(40.0, 20.0, 20.0, 60.0)))
        );
        let child_plan = capture
            .record_damage(DamageRegion::new(
                child,
                rect(45.0, 25.0, 5.0, 5.0),
                DamageReason::Hover,
            ))
            .expect("enabled capture records child damage")
            .clone();
        assert!(child_plan.subtracted_children.is_empty());
        let snapshot = capture.snapshot();
        assert_eq!(snapshot.parent_damage_subtractions, 1);
        assert_eq!(snapshot.damage.len(), 2);
        assert_eq!(snapshot.damage[1].damage.root, child);
    }

    #[test]
    fn clip_resize_and_reveal_are_visible_without_global_damage() {
        let mut capture = DebugCapture::enabled();
        let root = ScrollRootId::ROOT;
        capture.record_root(DebugRootInput {
            id: root,
            parent: None,
            viewport: rect(0.0, 0.0, 150.0, 100.0),
            clip: rect(0.0, 0.0, 150.0, 100.0),
            transform: LayerTransform::translated(-100.0, -20.0),
            grid: Some(grid()),
            content_viewport: Some(rect(100.0, 20.0, 150.0, 100.0)),
            visible_tiles: &[TileCoord::new(1, 0), TileCoord::new(2, 0)],
            resident_tiles: &[
                TileCoord::new(0, 0),
                TileCoord::new(1, 0),
                TileCoord::new(2, 0),
            ],
            newly_exposed_tiles: &[TileCoord::new(2, 0)],
        });
        capture.record_damage(DamageRegion {
            root,
            content_rect: rect(200.0, 0.0, 100.0, 100.0),
            reason: DamageReason::ScrollReveal,
        });
        capture.record_damage(DamageRegion {
            root,
            content_rect: rect(0.0, 0.0, 150.0, 100.0),
            reason: DamageReason::Clip,
        });
        let snapshot = capture.snapshot();
        assert_eq!(snapshot.damage.len(), 2);
        assert_eq!(snapshot.damage[0].damage.reason, DamageReason::ScrollReveal);
        assert_eq!(snapshot.damage[1].damage.reason, DamageReason::Clip);
        assert_eq!(snapshot.tiles.iter().filter(|tile| tile.visible).count(), 2);
        assert_eq!(snapshot.parent_damage_subtractions, 0);
    }
}
