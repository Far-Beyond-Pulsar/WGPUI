use crate::boundary::policy::{Pixels, Size};
use crate::geometry::Rect;
use crate::scene::layer::BoundaryId;
use crate::scene::tile::{EvictedTile, TileCoord, TileGrid, TileResidency};
use std::collections::HashMap;

/// Stable identity for a retained scroll root.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScrollRootId(u64);

impl ScrollRootId {
    pub const ROOT: Self = Self(0);

    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn from_boundary(boundary: BoundaryId) -> Self {
        Self(boundary.as_raw())
    }

    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// A tile address that cannot alias a tile belonging to another root or a
/// previous tile-grid configuration.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HierarchicalTileKey {
    pub root: ScrollRootId,
    pub tile: TileCoord,
    pub generation: u64,
}

/// Configuration for one retained scroll root.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScrollRootConfig {
    pub parent: Option<ScrollRootId>,
    pub viewport: Rect,
    pub content_origin: [f32; 2],
    pub content_extent: [f32; 2],
    pub tile_size: Option<Size<Pixels>>,
    pub retain_radius: u32,
    pub resident_tile_budget: usize,
    pub evict_after_frames: u32,
}

impl ScrollRootConfig {
    pub fn untiled(parent: Option<ScrollRootId>, viewport: Rect) -> Self {
        Self {
            parent,
            viewport,
            content_origin: [0.0; 2],
            content_extent: [viewport.width(), viewport.height()],
            tile_size: None,
            retain_radius: 0,
            resident_tile_budget: 1,
            evict_after_frames: 0,
        }
    }

    pub fn tiled(
        parent: Option<ScrollRootId>,
        viewport: Rect,
        tile_size: Size<Pixels>,
        retain_radius: u32,
        resident_tile_budget: usize,
    ) -> Self {
        Self {
            parent,
            viewport,
            content_origin: [0.0; 2],
            content_extent: [viewport.width(), viewport.height()],
            tile_size: Some(tile_size),
            retain_radius,
            resident_tile_budget,
            evict_after_frames: 60,
        }
    }
}

/// Why a region is dirty.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DamageReason {
    Content,
    Hover,
    Clip,
    ScrollReveal,
    Resource,
}

/// A dirty region owned by one scroll root.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Damage {
    pub root: ScrollRootId,
    pub region: Rect,
    pub reason: DamageReason,
}

/// The result of changing a root's viewport without rebuilding its content.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ClipChange {
    pub previous: Rect,
    pub current: Rect,
    pub damage: Damage,
}

/// The result of visiting a tiled root for a frame.
#[derive(Clone, Debug, PartialEq)]
pub struct ScrollRootVisit {
    pub root: ScrollRootId,
    pub grid: TileGrid,
    pub content_viewport: Rect,
    pub visible: Vec<TileCoord>,
    pub revealed: Vec<TileCoord>,
    pub evicted: Vec<EvictedTile>,
    pub over_budget: usize,
}

#[derive(Clone, Debug)]
struct ScrollRootState {
    config: ScrollRootConfig,
    current_offset: [f32; 2],
    previous_offset: [f32; 2],
    generation: u64,
    residency: Option<TileResidency>,
}

impl ScrollRootState {
    fn new(config: ScrollRootConfig, generation: u64) -> Self {
        Self {
            config,
            current_offset: [0.0; 2],
            previous_offset: [0.0; 2],
            generation,
            residency: None,
        }
    }

    fn grid(&self) -> Option<TileGrid> {
        self.config.tile_size.and_then(TileGrid::new)
    }

    fn content_viewport(&self) -> Rect {
        Rect::from_origin_size(
            [
                self.config.viewport.min_x - self.config.content_origin[0] - self.current_offset[0],
                self.config.viewport.min_y - self.config.content_origin[1] - self.current_offset[1],
            ],
            [self.config.viewport.width(), self.config.viewport.height()],
        )
    }
}

/// Retained hierarchy for independently-owned nested scroll roots.
#[derive(Clone, Debug, Default)]
pub struct ScrollRootTable {
    roots: HashMap<ScrollRootId, ScrollRootState>,
    next_generation: u64,
}

impl ScrollRootTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, root: ScrollRootId, config: ScrollRootConfig) -> bool {
        if config
            .parent
            .is_some_and(|parent| parent != root && !self.roots.contains_key(&parent))
        {
            return false;
        }
        let generation = self.next_generation();
        self.roots
            .insert(root, ScrollRootState::new(config, generation));
        true
    }

    pub fn remove(&mut self, root: ScrollRootId) -> bool {
        self.roots.remove(&root).is_some()
    }

    pub fn config(&self, root: ScrollRootId) -> Option<ScrollRootConfig> {
        self.roots.get(&root).map(|state| state.config)
    }

    pub fn generation(&self, root: ScrollRootId) -> Option<u64> {
        self.roots.get(&root).map(|state| state.generation)
    }

    pub fn set_offset(&mut self, root: ScrollRootId, offset: [f32; 2]) -> Option<[f32; 2]> {
        let state = self.roots.get_mut(&root)?;
        state.previous_offset = state.current_offset;
        state.current_offset = offset;
        Some([
            offset[0] - state.previous_offset[0],
            offset[1] - state.previous_offset[1],
        ])
    }

    pub fn offset(&self, root: ScrollRootId) -> Option<[f32; 2]> {
        self.roots.get(&root).map(|state| state.current_offset)
    }

    pub fn update_viewport(&mut self, root: ScrollRootId, viewport: Rect) -> Option<ClipChange> {
        let state = self.roots.get_mut(&root)?;
        let previous = state.config.viewport;
        if previous == viewport {
            return None;
        }
        state.config.viewport = viewport;
        Some(ClipChange {
            previous,
            current: viewport,
            damage: Damage {
                root,
                region: previous.union(&viewport),
                reason: DamageReason::Clip,
            },
        })
    }

    pub fn visit(&mut self, root: ScrollRootId, frame: u64) -> Option<ScrollRootVisit> {
        let state = self.roots.get_mut(&root)?;
        let grid = state.grid()?;
        let content_viewport = state.content_viewport();
        let span = grid.visible_span(content_viewport, state.config.retain_radius)?;
        let residency = state
            .residency
            .get_or_insert_with(|| TileResidency::new(state.config.resident_tile_budget));
        residency.set_budget(state.config.resident_tile_budget);
        let visible = span.tiles();
        let revealed = residency.mark(span, frame);
        let evicted = residency.sweep(frame, state.config.evict_after_frames);
        Some(ScrollRootVisit {
            root,
            grid,
            content_viewport,
            visible,
            revealed,
            evicted,
            over_budget: residency.over_budget(),
        })
    }

    pub fn tile_key(&self, root: ScrollRootId, tile: TileCoord) -> Option<HierarchicalTileKey> {
        let state = self.roots.get(&root)?;
        state.grid()?;
        Some(HierarchicalTileKey {
            root,
            tile,
            generation: state.generation,
        })
    }

    pub fn is_current(&self, key: HierarchicalTileKey) -> bool {
        self.tile_key(key.root, key.tile) == Some(key)
    }

    pub fn tile_bounds(&self, key: HierarchicalTileKey) -> Option<Rect> {
        let state = self.roots.get(&key.root)?;
        if state.generation != key.generation {
            return None;
        }
        let grid = state.grid()?;
        let bounds = grid.tile_bounds(key.tile);
        Some(Rect::from_origin_size(
            [
                bounds.min_x + state.config.content_origin[0] + state.current_offset[0],
                bounds.min_y + state.config.content_origin[1] + state.current_offset[1],
            ],
            [bounds.width(), bounds.height()],
        ))
    }

    pub fn tile_keys_for_region(
        &self,
        root: ScrollRootId,
        region: Rect,
    ) -> Vec<HierarchicalTileKey> {
        let Some(state) = self.roots.get(&root) else {
            return Vec::new();
        };
        let Some(grid) = state.grid() else {
            return Vec::new();
        };
        let content_region = Rect::from_origin_size(
            [
                region.min_x - state.config.content_origin[0] - state.current_offset[0],
                region.min_y - state.config.content_origin[1] - state.current_offset[1],
            ],
            [region.width(), region.height()],
        );
        let Some(span) = grid.span(content_region) else {
            return Vec::new();
        };
        span.tiles()
            .into_iter()
            .filter_map(|tile| self.tile_key(root, tile))
            .collect()
    }

    pub fn hover_damage(&self, root: ScrollRootId, previous: Rect, current: Rect) -> Damage {
        Damage {
            root,
            region: previous.union(&current),
            reason: DamageReason::Hover,
        }
    }

    pub fn parent_damage(&self, damage: Damage) -> Vec<Damage> {
        let mut regions = vec![damage.region];
        for child in self.descendants(damage.root) {
            let Some(state) = self.roots.get(&child) else {
                continue;
            };
            regions = regions
                .into_iter()
                .flat_map(|region| subtract_rect(region, state.config.viewport))
                .collect();
        }
        regions
            .into_iter()
            .filter(|region| !region.is_empty())
            .map(|region| Damage { region, ..damage })
            .collect()
    }

    fn descendants(&self, root: ScrollRootId) -> Vec<ScrollRootId> {
        let mut descendants = Vec::new();
        let mut pending = vec![root];
        while let Some(parent) = pending.pop() {
            for (candidate, state) in &self.roots {
                if state.config.parent == Some(parent) {
                    descendants.push(*candidate);
                    pending.push(*candidate);
                }
            }
        }
        descendants.sort_unstable();
        descendants
    }

    fn next_generation(&mut self) -> u64 {
        self.next_generation = self.next_generation.wrapping_add(1);
        self.next_generation
    }
}

fn subtract_rect(region: Rect, cover: Rect) -> Vec<Rect> {
    let intersection = region.intersect(&cover);
    if intersection.is_empty() {
        return vec![region];
    }
    let mut result = Vec::with_capacity(4);
    if region.min_x < intersection.min_x {
        result.push(Rect {
            min_x: region.min_x,
            min_y: region.min_y,
            max_x: intersection.min_x,
            max_y: region.max_y,
        });
    }
    if intersection.max_x < region.max_x {
        result.push(Rect {
            min_x: intersection.max_x,
            min_y: region.min_y,
            max_x: region.max_x,
            max_y: region.max_y,
        });
    }
    if region.min_y < intersection.min_y {
        result.push(Rect {
            min_x: intersection.min_x,
            min_y: region.min_y,
            max_x: intersection.max_x,
            max_y: intersection.min_y,
        });
    }
    if intersection.max_y < region.max_y {
        result.push(Rect {
            min_x: intersection.min_x,
            min_y: intersection.max_y,
            max_x: intersection.max_x,
            max_y: region.max_y,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiled(_root: ScrollRootId, parent: Option<ScrollRootId>) -> ScrollRootConfig {
        ScrollRootConfig::tiled(
            parent,
            Rect::from_origin_size([0.0, 0.0], [100.0, 100.0]),
            Size::pixels(50.0, 50.0),
            0,
            16,
        )
    }

    #[test]
    fn roots_own_same_coordinate_independently() {
        let mut table = ScrollRootTable::new();
        assert!(table.insert(
            ScrollRootId::from_raw(1),
            tiled(ScrollRootId::from_raw(1), None)
        ));
        assert!(table.insert(
            ScrollRootId::from_raw(2),
            tiled(ScrollRootId::from_raw(2), None)
        ));
        let first = table.tile_key(ScrollRootId::from_raw(1), TileCoord::ORIGIN);
        let second = table.tile_key(ScrollRootId::from_raw(2), TileCoord::ORIGIN);
        assert_ne!(first, second);
    }

    #[test]
    fn invalid_tile_size_falls_back_without_losing_the_root() {
        let root = ScrollRootId::from_raw(1);
        let mut table = ScrollRootTable::new();
        let mut config =
            ScrollRootConfig::untiled(None, Rect::from_origin_size([0.0, 0.0], [10.0, 10.0]));
        config.tile_size = Some(Size::pixels(0.0, 10.0));
        assert!(table.insert(root, config));
        assert!(table.visit(root, 1).is_none());
        assert!(table.config(root).is_some());
    }

    #[test]
    fn crossing_a_tile_boundary_reveals_only_new_tiles() {
        let root = ScrollRootId::from_raw(1);
        let mut table = ScrollRootTable::new();
        assert!(table.insert(root, tiled(root, None)));
        let first = table.visit(root, 1).expect("tiled visit");
        assert_eq!(first.revealed.len(), 4);
        table.set_offset(root, [-50.0, 0.0]);
        let second = table.visit(root, 2).expect("tiled visit");
        assert_eq!(second.revealed.len(), 2);
    }

    #[test]
    fn parent_damage_excludes_nested_child_pixels() {
        let parent = ScrollRootId::from_raw(1);
        let child = ScrollRootId::from_raw(2);
        let mut table = ScrollRootTable::new();
        assert!(table.insert(
            parent,
            ScrollRootConfig::untiled(None, Rect::from_origin_size([0.0, 0.0], [100.0, 100.0]))
        ));
        assert!(table.insert(
            child,
            ScrollRootConfig::untiled(
                Some(parent),
                Rect::from_origin_size([25.0, 25.0], [50.0, 50.0])
            )
        ));
        let damage = Damage {
            root: parent,
            region: Rect::from_origin_size([0.0, 0.0], [100.0, 100.0]),
            reason: DamageReason::Content,
        };
        let regions = table.parent_damage(damage);
        assert_eq!(regions.len(), 4);
        assert!(regions.iter().all(|region| {
            !region
                .region
                .intersects(&Rect::from_origin_size([25.0, 25.0], [50.0, 50.0]))
        }));
    }

    #[test]
    fn hover_damage_is_the_union_and_resize_is_clip_only() {
        let root = ScrollRootId::from_raw(1);
        let mut table = ScrollRootTable::new();
        assert!(table.insert(
            root,
            ScrollRootConfig::untiled(None, Rect::from_origin_size([0.0, 0.0], [100.0, 100.0]))
        ));
        let hover = table.hover_damage(
            root,
            Rect::from_origin_size([0.0, 0.0], [10.0, 10.0]),
            Rect::from_origin_size([20.0, 20.0], [10.0, 10.0]),
        );
        assert_eq!(hover.reason, DamageReason::Hover);
        let change = table
            .update_viewport(root, Rect::from_origin_size([0.0, 0.0], [80.0, 100.0]))
            .expect("changed");
        assert_eq!(change.damage.reason, DamageReason::Clip);
        assert_eq!(change.damage.region.max_x, 100.0);
    }
}
