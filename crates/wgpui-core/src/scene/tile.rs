//! `TileCoord`, the tile plane's geometry, the visibility predicate, and the
//! residency budget behind `Buffering::Tiled`.
//! See docs/gpu-native-architecture.md §4.3.
//!
//! # A tile is a `Layer`, and that is the whole design
//!
//! §4.3 opens with the observation this file is built on: "a tile is just a
//! `Layer`, addressed one dimension further." Everything a tile needs — its own
//! instance arena, its own slab, its own place in occlusion culling and indirect
//! draw issuance — is what a [`crate::scene::layer::Layer`] already is, so
//! nothing here is a parallel tile-specific residency structure. What is here is
//! only what a `Layer` cannot answer for itself:
//!
//! - **Where a tile sits on the content plane** ([`TileGrid`]), which is integer
//!   arithmetic over a fixed edge length and nothing more.
//! - **Which tiles are in range right now** ([`TileGrid::visible_span`]), which
//!   is §4.3's "which tile coordinates intersect (viewport ∪ retain radius) at
//!   the current pan offset" — the predicate `shaders/tile_visibility.wgsl`
//!   transcribes and `wgpui-wgpu`'s differential checks for exact equality.
//! - **Which tiles stay resident** ([`TileResidency`]), which is R-N §3.4's
//!   mark-and-sweep triggered spatially instead of by visit, plus the total
//!   resident-tile budget §4.3 and §9's risk table both call for.
//!
//! # Phase 1 defined the address; Phase 4.5 defines the mechanism
//!
//! [`TileCoord`] and [`crate::scene::layer::LayerKey::tiled`] shipped in Phase 1
//! deliberately unused, so that this phase extends an addressing scheme rather
//! than reshaping the identity of every layer in a shipped scene. Nothing about
//! `LayerKey` changed here, which is the evidence that the bet paid.

use crate::boundary::policy::{Pixels, Size};
use crate::geometry::Rect;
use crate::scene::layer::LayerTransform;
use std::collections::HashMap;

/// A tile's integer coordinate on its boundary's content plane.
///
/// Coordinates are signed because a freely-pannable plane has no origin corner
/// — panning up and left from the initial view produces negative coordinates,
/// and clamping them at zero would silently alias two distinct tiles onto one
/// address.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileCoord {
    /// Column index along the horizontal axis.
    pub x: i32,
    /// Row index along the vertical axis.
    pub y: i32,
}

impl TileCoord {
    /// The tile containing the plane's origin.
    pub const ORIGIN: TileCoord = TileCoord { x: 0, y: 0 };

    /// A coordinate at `(x, y)`.
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// An inclusive rectangle of tile coordinates.
///
/// Inclusive rather than half-open because it is produced by rounding a float
/// rectangle outward on both edges, and a half-open form would need one of the
/// two roundings to be spelled differently from the other — which is exactly
/// where an off-by-one lives.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileSpan {
    /// Top-left tile, inclusive.
    pub min: TileCoord,
    /// Bottom-right tile, inclusive.
    pub max: TileCoord,
}

impl TileSpan {
    /// The largest tile count [`TileGrid::visible_span`] will produce.
    ///
    /// Not a tuning knob — a guard. A span is derived from a viewport divided by
    /// an author-chosen tile size, so an author who asks for 1px tiles on a 4K
    /// viewport asks for eight million layers. Every real configuration is far
    /// under this: a 3840×2160 viewport with 256px tiles and a retain radius of
    /// 2 is 19×13 = 247 tiles. A span that would exceed it is reported as
    /// unusable rather than materialized, so the caller falls back to untiled
    /// buffering instead of the process dying.
    pub const MAX_TILES: u64 = 4_096;

    /// The span holding exactly one tile.
    pub const fn single(coord: TileCoord) -> TileSpan {
        TileSpan {
            min: coord,
            max: coord,
        }
    }

    /// How many tiles this span covers.
    ///
    /// `u64` because the two coordinates are `i32`s the caller chose and their
    /// difference does not fit in one.
    pub fn tile_count(&self) -> u64 {
        let width = i64::from(self.max.x) - i64::from(self.min.x) + 1;
        let height = i64::from(self.max.y) - i64::from(self.min.y) + 1;
        if width <= 0 || height <= 0 {
            return 0;
        }
        (width as u64).saturating_mul(height as u64)
    }

    /// Whether `coord` lies in this span.
    pub fn contains(&self, coord: TileCoord) -> bool {
        coord.x >= self.min.x
            && coord.x <= self.max.x
            && coord.y >= self.min.y
            && coord.y <= self.max.y
    }

    /// Every tile in the span, row-major and ascending — a deterministic order,
    /// so a caller that turns this into layers gets the same layer order every
    /// frame.
    pub fn tiles(&self) -> Vec<TileCoord> {
        let mut tiles = Vec::with_capacity(self.tile_count().min(TileSpan::MAX_TILES) as usize);
        let mut y = self.min.y;
        while y <= self.max.y {
            let mut x = self.min.x;
            while x <= self.max.x {
                tiles.push(TileCoord::new(x, y));
                if x == i32::MAX {
                    break;
                }
                x += 1;
            }
            if y == i32::MAX {
                break;
            }
            y += 1;
        }
        tiles
    }

    /// This span grown by `radius` tiles on every side, saturating at the
    /// coordinate space's edges.
    pub fn expanded(&self, radius: u32) -> TileSpan {
        let radius = i32::try_from(radius).unwrap_or(i32::MAX);
        TileSpan {
            min: TileCoord::new(
                self.min.x.saturating_sub(radius),
                self.min.y.saturating_sub(radius),
            ),
            max: TileCoord::new(
                self.max.x.saturating_add(radius),
                self.max.y.saturating_add(radius),
            ),
        }
    }
}

/// Where a boundary's tiles sit on its content plane.
///
/// Holds one thing — the tile edge length — because that is the only degree of
/// freedom §4.3 grants. Multi-resolution tiling is explicitly rejected (§10), so
/// there is no scale here and a zoom change re-renders at the new scale exactly
/// as `Buffering::Margin` does.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TileGrid {
    width: f32,
    height: f32,
}

impl TileGrid {
    /// The smallest tile edge that can produce a usable grid.
    ///
    /// Below a pixel a "tile" addresses less than one sample, and the span
    /// arithmetic stops being meaningful before it stops being finite.
    pub const MIN_EDGE: f32 = 1.0;

    /// The starting tile edge, in logical pixels.
    ///
    /// §4.3 asks for "a starting size from common compositor practice (roughly
    /// 256–512px)" *validated against a representative node-graph workload, not
    /// asserted*. 256 is the low end of that range, and
    /// `wgpui-wgpu/examples/phase45_tiling_bench.rs` is the validation — see
    /// `docs/phase-4.5-results.md` for the measured sweep this number came out
    /// of, which is the reason it is 256 and not 512.
    pub const DEFAULT_EDGE: f32 = 256.0;

    /// A grid of `tile_size` tiles, or `None` if that size cannot address a
    /// plane.
    ///
    /// Fallible rather than clamping: a caller that asked for a zero or negative
    /// tile size has a bug, and silently substituting a working size would hide
    /// it behind a grid that quietly is not the one requested.
    pub fn new(tile_size: Size<Pixels>) -> Option<TileGrid> {
        let width = tile_size.width.value();
        let height = tile_size.height.value();
        if !(width >= TileGrid::MIN_EDGE) || !(height >= TileGrid::MIN_EDGE) {
            return None;
        }
        Some(TileGrid { width, height })
    }

    /// A square grid of `edge`-pixel tiles.
    pub fn square(edge: f32) -> Option<TileGrid> {
        TileGrid::new(Size::pixels(edge, edge))
    }

    /// The tile edge lengths.
    pub fn tile_size(&self) -> Size<Pixels> {
        Size::pixels(self.width, self.height)
    }

    /// Where `coord`'s tile sits on the content plane.
    pub fn tile_bounds(&self, coord: TileCoord) -> Rect {
        let min_x = coord.x as f32 * self.width;
        let min_y = coord.y as f32 * self.height;
        Rect {
            min_x,
            min_y,
            max_x: min_x + self.width,
            max_y: min_y + self.height,
        }
    }

    /// The tile containing a point on the content plane.
    pub fn containing(&self, point: [f32; 2]) -> TileCoord {
        TileCoord::new(
            floor_to_i32(point[0] / self.width),
            floor_to_i32(point[1] / self.height),
        )
    }

    /// The tiles `region` intersects, or `None` when it encloses no area.
    ///
    /// **The edge rule matches [`Rect::intersects`] exactly**, and that is
    /// load-bearing rather than incidental: a region whose right edge lands
    /// exactly on a tile boundary does not reach into the next tile, the same
    /// way two rectangles that merely touch along an edge do not intersect
    /// anywhere else in this crate. `shaders/tile_visibility.wgsl` gets the same
    /// answer by testing rectangles directly rather than by dividing, and
    /// `visible_span_agrees_with_a_direct_rectangle_test` pins the two
    /// formulations together.
    pub fn span(&self, region: Rect) -> Option<TileSpan> {
        if region.is_empty() {
            return None;
        }
        Some(TileSpan {
            min: TileCoord::new(
                floor_to_i32(region.min_x / self.width),
                floor_to_i32(region.min_y / self.height),
            ),
            max: TileCoord::new(
                ceil_to_i32(region.max_x / self.width).saturating_sub(1),
                ceil_to_i32(region.max_y / self.height).saturating_sub(1),
            ),
        })
    }

    /// The content-plane rectangle currently under `viewport`, given where the
    /// boundary's layers composite.
    ///
    /// A layer transform displaces content, so the plane slides under a fixed
    /// window by the negative of it. Spelled as its own function rather than
    /// inlined at the two call sites because getting its sign wrong produces a
    /// grid that pans the wrong way — visibly, but only once a real window
    /// exists to see it in.
    pub fn content_viewport(viewport: Rect, transform: LayerTransform) -> Rect {
        let [x, y] = transform.translation;
        Rect {
            min_x: viewport.min_x - x,
            min_y: viewport.min_y - y,
            max_x: viewport.max_x - x,
            max_y: viewport.max_y - y,
        }
    }

    /// §4.3's visibility set: the tiles intersecting (viewport ∪ retain radius)
    /// at the current pan offset.
    ///
    /// `None` when the viewport encloses no area, or when this tile size would
    /// put more than [`TileSpan::MAX_TILES`] tiles in range — see that constant
    /// for why the second case is reported rather than materialized.
    pub fn visible_span(&self, content_viewport: Rect, retain_radius: u32) -> Option<TileSpan> {
        let span = self.span(content_viewport)?.expanded(retain_radius);
        if span.tile_count() > TileSpan::MAX_TILES {
            return None;
        }
        Some(span)
    }

    /// Which layer a primitive with these bounds belongs in.
    pub fn placement(&self, bounds: Rect) -> TilePlacement {
        match self.span(bounds) {
            Some(span) if span.tile_count() == 1 => TilePlacement::Tile(span.min),
            // A degenerate primitive has no tile to be inside of, and putting it
            // on the overlay costs one never-visible instance rather than
            // needing a third case that means "nowhere".
            _ => TilePlacement::Overlay,
        }
    }
}

/// Which of a tiled boundary's layers a primitive is emitted into.
///
/// # The multi-tile content rule, chosen and stated once
///
/// §4.3 offers two: clip a spanning primitive into each tile it crosses (what
/// browser tiling does), or put it on an unbuffered overlay layer above the
/// grid, "the same named pattern SFD §2 already proposes for hover-resolved
/// content that can't cleanly live inside a buffer," with the instruction to
/// "reuse that pattern rather than inventing a second one."
///
/// **This picks the overlay**, for a reason that is about what exists rather
/// than about taste: per-tile clipping needs a per-primitive clip rectangle, and
/// [`crate::patch::primitive::Quad`] does not have one — `docs/phase-1-results.md`
/// §2 already recorded that absence. Clipping *geometrically* instead is not the
/// same operation: shrinking a rounded, bordered quad's rectangle moves its
/// corners and its border inward, so a node body straddling a tile edge would
/// render as two differently-shaped halves. So clipping is not merely more work
/// here, it is not yet expressible, and the version that is expressible is
/// wrong.
///
/// The overlay costs nothing new: it is [`crate::scene::layer::LayerKey::untiled`],
/// the same layer an untiled boundary already uses, and it pans by `TRANSFORM`
/// with the tiles. Its honest price is in `docs/phase-4.5-results.md`: overlay
/// content is not tile-culled, so a graph whose wires are mostly long puts a
/// growing always-resident layer above a well-bounded grid. Per-tile clipping
/// becomes the better answer the moment `Quad` gains a content mask, and that is
/// where the rule should be revisited.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TilePlacement {
    /// The primitive fits inside one tile and lives in that tile's layer.
    Tile(TileCoord),
    /// The primitive spans more than one tile and lives on the boundary's
    /// unbuffered overlay layer.
    Overlay,
}

/// One resident tile's bookkeeping.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileResidencyState {
    /// The last frame this tile was in range.
    pub last_visited_frame: u64,
    /// A monotonic stamp, assigned in visit order.
    ///
    /// The LRU order is taken from this rather than from `last_visited_frame`
    /// because a whole span shares one frame number, and an eviction order that
    /// ties across a hundred tiles would be decided by hash iteration order —
    /// which is to say, not decided. This makes it total and reproducible.
    pub last_touch: u64,
}

/// Why a tile stopped being resident.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileEviction {
    /// It sat outside (viewport ∪ retain radius) for longer than
    /// `evict_after_frames` — R-N §3.4's mark-and-sweep, triggered spatially.
    OutOfRange,
    /// It was the least recently visited tile still over the budget.
    Budget,
}

/// One tile leaving residency.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EvictedTile {
    /// Which tile.
    pub coord: TileCoord,
    /// Why.
    pub reason: TileEviction,
}

/// Which tiles of one boundary are resident, and what leaves.
///
/// # Two mechanisms, because one was never enough
///
/// R-N §3.4's mark-and-sweep is a per-item timer, and §4.3 says plainly why it
/// does not bound a tile grid on its own: "an erratic pan pattern (fast diagonal
/// movement, frequent direction reversal) can keep many tiles within 'recently
/// visited' simultaneously … a freeform 2D plane can have far more live tiles
/// than a typical UI ever had layers." So there are two:
///
/// 1. [`TileResidency::sweep`] evicts tiles out of range for longer than the
///    boundary's `evict_after_frames`. This is R-N's mechanism unchanged, with
///    "unvisited" meaning "not in range" instead of "not in the tree".
/// 2. The same call then evicts least-recently-visited tiles until the resident
///    count is inside `budget`.
///
/// **A tile in range this frame is never evicted by either.** The budget cannot
/// be honoured against a visible set larger than itself, and evicting a tile the
/// next line of the frame is about to re-render would trade a memory bound for
/// unbounded work. [`TileResidency::over_budget`] reports that state instead of
/// hiding it.
#[derive(Clone, Debug)]
pub struct TileResidency {
    tiles: HashMap<TileCoord, TileResidencyState>,
    budget: usize,
    next_touch: u64,
    over_budget: usize,
}

impl TileResidency {
    /// A residency holding nothing, capped at `budget` tiles.
    pub fn new(budget: usize) -> TileResidency {
        TileResidency {
            tiles: HashMap::new(),
            budget: budget.max(1),
            next_touch: 0,
            over_budget: 0,
        }
    }

    /// Mark every tile of `span` in range as of `frame`, returning the ones that
    /// were not resident before.
    ///
    /// Those are the newly-revealed tiles — §4.3's "crossing into a new tile
    /// triggers `DISPLAY` for *that tile alone*". The caller turns each into a
    /// `LayerKey::tiled` layer, and a brand-new layer starts fully invalidated
    /// by [`crate::scene::layer::LayerTable::insert`]'s own rule, so nothing
    /// here has to special-case a tile into being dirty.
    ///
    /// Returned sorted, so a frame's newly-revealed set is a reproducible list
    /// rather than a hash order.
    pub fn mark(&mut self, span: TileSpan, frame: u64) -> Vec<TileCoord> {
        let mut revealed = Vec::new();
        for coord in span.tiles() {
            self.next_touch = self.next_touch.wrapping_add(1);
            let touch = self.next_touch;
            match self.tiles.get_mut(&coord) {
                Some(state) => {
                    state.last_visited_frame = frame;
                    state.last_touch = touch;
                }
                None => {
                    self.tiles.insert(
                        coord,
                        TileResidencyState {
                            last_visited_frame: frame,
                            last_touch: touch,
                        },
                    );
                    revealed.push(coord);
                }
            }
        }
        revealed.sort_unstable();
        revealed
    }

    /// Evict everything out of range for too long, then everything over budget.
    ///
    /// Returns what left, sorted, each with the rule that removed it. See this
    /// type's doc for why a tile visited on `frame` survives both rules.
    pub fn sweep(&mut self, frame: u64, evict_after_frames: u32) -> Vec<EvictedTile> {
        let mut evicted = Vec::new();
        self.tiles.retain(|coord, state| {
            if state.last_visited_frame >= frame {
                return true;
            }
            let elapsed = frame.saturating_sub(state.last_visited_frame);
            if elapsed <= u64::from(evict_after_frames) {
                return true;
            }
            evicted.push(EvictedTile {
                coord: *coord,
                reason: TileEviction::OutOfRange,
            });
            false
        });

        // LRU beyond the budget. Candidates are only tiles not in range this
        // frame, so the loop terminates whether or not the budget can be met.
        let mut candidates: Vec<(u64, TileCoord)> = self
            .tiles
            .iter()
            .filter(|(_, state)| state.last_visited_frame < frame)
            .map(|(coord, state)| (state.last_touch, *coord))
            .collect();
        candidates.sort_unstable();
        let mut over = self.tiles.len().saturating_sub(self.budget);
        for (_, coord) in candidates {
            if over == 0 {
                break;
            }
            if self.tiles.remove(&coord).is_some() {
                evicted.push(EvictedTile {
                    coord,
                    reason: TileEviction::Budget,
                });
                over -= 1;
            }
        }
        self.over_budget = over;

        evicted.sort_unstable_by_key(|entry| entry.coord);
        evicted
    }

    /// How many tiles the budget could not account for after the last
    /// [`TileResidency::sweep`], because they were all in range.
    ///
    /// Non-zero means the visible set alone is larger than the budget, i.e. the
    /// tile size is too small or the budget too tight for this viewport. Not an
    /// error and not silently absorbed: a caller that wants to know it is paying
    /// more than it asked for can read it.
    pub const fn over_budget(&self) -> usize {
        self.over_budget
    }

    /// The resident-tile cap.
    pub const fn budget(&self) -> usize {
        self.budget
    }

    /// Whether this tile is resident.
    pub fn contains(&self, coord: TileCoord) -> bool {
        self.tiles.contains_key(&coord)
    }

    /// One tile's bookkeeping.
    pub fn state(&self, coord: TileCoord) -> Option<TileResidencyState> {
        self.tiles.get(&coord).copied()
    }

    /// Every resident tile, sorted.
    pub fn resident(&self) -> Vec<TileCoord> {
        let mut tiles: Vec<TileCoord> = self.tiles.keys().copied().collect();
        tiles.sort_unstable();
        tiles
    }

    /// How many tiles are resident.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Whether no tile is resident.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }
}

/// One resident tile's descriptor, as `shaders/tile_visibility.wgsl` reads it.
///
/// Carries the tile's coordinate *and* its slab reservation, because the pass's
/// whole job is to turn the first into a decision about the second without the
/// CPU deciding which tiles draw.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileDescriptor {
    /// Which tile.
    pub coord: TileCoord,
    /// Its layer's reservation base in the kind's arena.
    pub base: u32,
    /// Slots reserved there.
    pub count: u32,
}

/// Bytes one [`TileDescriptor`] occupies on the device: `[x, y, base, count]`
/// as a `vec4<u32>`, with the coordinate's two `i32`s bit-cast in place.
pub const TILE_DESCRIPTOR_STRIDE: usize = 16;

/// Encode tile descriptors for `shaders/tile_visibility.wgsl`.
///
/// Byte-oriented for the reason `patch/primitive.rs` gives: it keeps
/// `wgpui-core` dependency-free and makes the GPU layout an explicit decision.
pub fn encode_tiles(tiles: &[TileDescriptor], destination: &mut Vec<u8>) {
    destination.clear();
    destination.reserve(tiles.len() * TILE_DESCRIPTOR_STRIDE);
    for tile in tiles {
        for value in [
            tile.coord.x as u32,
            tile.coord.y as u32,
            tile.base,
            tile.count,
        ] {
            destination.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// What the tile-visibility pass decides, as plain data.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileVisibility {
    /// One `[base, count, 0, 0]` record per tile, in the order the descriptors
    /// were given — exactly [`crate::indirect::encode_slots`]' layout, because
    /// this *is* the slot table the indirect-arg pass then consumes.
    ///
    /// An out-of-range tile's `count` is zero, so Phase 4's `compact` writes it
    /// a zero-instance argument record and `pack` drops it. Nothing new draws;
    /// the existing mechanism simply finds nothing to draw for that tile.
    pub slots: Vec<[u32; 4]>,
    /// One flag per tile: `1` in range, `0` out. Not read by the draw path — it
    /// exists so the differential can say *which* tile disagreed rather than
    /// only that a slot did.
    pub in_range: Vec<u32>,
}

impl TileVisibility {
    /// How many tiles were in range.
    pub fn visible_count(&self) -> usize {
        self.in_range.iter().filter(|flag| **flag != 0).count()
    }

    /// The slot records, encoded as [`crate::indirect::encode_slots`] would.
    pub fn encode_slots(&self, destination: &mut Vec<u8>) {
        destination.clear();
        destination.reserve(self.slots.len() * 16);
        for slot in &self.slots {
            for value in slot {
                destination.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
}

/// The CPU reference `shaders/tile_visibility.wgsl` transcribes.
///
/// For each descriptor, decide whether its tile intersects the content-space
/// viewport dilated by `retain_radius` tiles, and emit that tile's draw slot
/// with its real reservation count if so and zero if not.
///
/// **The dilation is exact, not approximate.** Growing the viewport rectangle by
/// `retain_radius × tile_size` on each side selects precisely the same tiles as
/// growing [`TileGrid::span`]'s result by `retain_radius` coordinates, because
/// tile edges sit at exact multiples of the tile size — so `floor((min - r·w)/w)`
/// *is* `floor(min/w) - r`. That equivalence is what lets the shader test
/// rectangles (cheap, branch-free, no division) while
/// [`TileResidency`] works in coordinates, and
/// `visible_span_agrees_with_a_direct_rectangle_test` asserts it rather than
/// trusting this paragraph.
pub fn tile_visibility(
    grid: &TileGrid,
    tiles: &[TileDescriptor],
    content_viewport: Rect,
    retain_radius: u32,
) -> TileVisibility {
    let margin_x = retain_radius as f32 * grid.width;
    let margin_y = retain_radius as f32 * grid.height;
    let range = Rect {
        min_x: content_viewport.min_x - margin_x,
        min_y: content_viewport.min_y - margin_y,
        max_x: content_viewport.max_x + margin_x,
        max_y: content_viewport.max_y + margin_y,
    };
    let mut slots = Vec::with_capacity(tiles.len());
    let mut in_range = Vec::with_capacity(tiles.len());
    for tile in tiles {
        let bounds = grid.tile_bounds(tile.coord);
        let visible = range.intersects(&bounds);
        in_range.push(u32::from(visible));
        slots.push([tile.base, if visible { tile.count } else { 0 }, 0, 0]);
    }
    TileVisibility { slots, in_range }
}

/// `f32::floor` narrowed to `i32`, saturating rather than wrapping.
///
/// `as` on an out-of-range float already saturates in Rust, and NaN becomes
/// zero; both are spelled out here because a coordinate that silently wrapped
/// would alias two distant tiles onto one address, which is the exact failure
/// [`TileCoord`] being signed exists to prevent.
fn floor_to_i32(value: f32) -> i32 {
    if value.is_nan() {
        return 0;
    }
    value.floor() as i32
}

fn ceil_to_i32(value: f32) -> i32 {
    if value.is_nan() {
        return 0;
    }
    value.ceil() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> TileGrid {
        TileGrid::square(256.0).expect("256px is a usable tile edge")
    }

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
        Rect::from_origin_size([x, y], [width, height])
    }

    #[test]
    fn negative_coordinates_are_distinct_addresses() {
        assert_ne!(TileCoord::new(-1, 0), TileCoord::new(1, 0));
        assert_ne!(TileCoord::new(0, -1), TileCoord::ORIGIN);
    }

    #[test]
    fn a_plane_has_no_origin_corner_so_panning_up_and_left_addresses_negatives() {
        let grid = grid();
        assert_eq!(grid.containing([-1.0, -1.0]), TileCoord::new(-1, -1));
        assert_eq!(grid.containing([0.0, 0.0]), TileCoord::ORIGIN);
        assert_eq!(grid.containing([-256.0, 0.0]), TileCoord::new(-1, 0));
        assert_eq!(grid.containing([-257.0, 0.0]), TileCoord::new(-2, 0));
        assert_eq!(
            grid.tile_bounds(TileCoord::new(-1, -2)),
            rect(-256.0, -512.0, 256.0, 256.0)
        );
    }

    #[test]
    fn a_region_ending_exactly_on_a_tile_edge_does_not_reach_into_the_next_tile() {
        let grid = grid();
        assert_eq!(
            grid.span(rect(0.0, 0.0, 512.0, 256.0)),
            Some(TileSpan {
                min: TileCoord::ORIGIN,
                max: TileCoord::new(1, 0),
            }),
            "the same strictness Rect::intersects has everywhere else in the crate"
        );
        assert_eq!(
            grid.span(rect(0.0, 0.0, 513.0, 256.0)).map(|s| s.max),
            Some(TileCoord::new(2, 0))
        );
        assert_eq!(grid.span(rect(10.0, 10.0, 0.0, 100.0)), None);
    }

    /// The equivalence `tile_visibility`'s doc rests on: dilating the rectangle
    /// by `radius × tile_size` picks exactly the tiles that dilating the span by
    /// `radius` coordinates picks. The shader does the first; residency does the
    /// second. If they ever disagree, a tile draws that nothing rendered into.
    #[test]
    fn visible_span_agrees_with_a_direct_rectangle_test() {
        let grid = grid();
        for radius in 0..3u32 {
            for step in 0..40 {
                let viewport = rect(
                    step as f32 * 37.0 - 400.0,
                    step as f32 * 23.0 - 300.0,
                    900.0,
                    600.0,
                );
                let span = grid
                    .visible_span(viewport, radius)
                    .expect("a 900x600 viewport on a 256px grid is usable");
                let descriptors: Vec<TileDescriptor> = span
                    .expanded(2)
                    .tiles()
                    .into_iter()
                    .map(|coord| TileDescriptor {
                        coord,
                        base: 0,
                        count: 1,
                    })
                    .collect();
                let visibility = tile_visibility(&grid, &descriptors, viewport, radius);
                for (tile, flag) in descriptors.iter().zip(&visibility.in_range) {
                    assert_eq!(
                        *flag != 0,
                        span.contains(tile.coord),
                        "radius {radius} step {step} disagreed on {:?}",
                        tile.coord
                    );
                }
            }
        }
    }

    #[test]
    fn a_retain_radius_buys_a_ring_of_tiles_on_every_side() {
        let grid = grid();
        let viewport = rect(0.0, 0.0, 256.0, 256.0);
        assert_eq!(
            grid.visible_span(viewport, 0).map(|s| s.tile_count()),
            Some(1)
        );
        assert_eq!(
            grid.visible_span(viewport, 1).map(|s| s.tile_count()),
            Some(9),
            "one ring around a single visible tile is 3x3"
        );
        assert_eq!(
            grid.visible_span(viewport, 2).map(|s| s.tile_count()),
            Some(25)
        );
    }

    #[test]
    fn panning_moves_the_content_viewport_against_the_transform() {
        let viewport = rect(0.0, 0.0, 800.0, 600.0);
        // The user drags the canvas left, so content composites at a negative x
        // and the plane under the window slides right.
        let panned = TileGrid::content_viewport(viewport, LayerTransform::translated(-300.0, 0.0));
        assert_eq!(panned.min_x, 300.0);
        assert_eq!(panned.max_x, 1100.0);
        assert_eq!(
            TileGrid::content_viewport(viewport, LayerTransform::IDENTITY),
            viewport
        );
    }

    #[test]
    fn an_unusable_tile_size_is_reported_rather_than_clamped() {
        assert!(TileGrid::new(Size::pixels(0.0, 256.0)).is_none());
        assert!(TileGrid::new(Size::pixels(256.0, -4.0)).is_none());
        assert!(TileGrid::new(Size::pixels(f32::NAN, 256.0)).is_none());
        assert!(TileGrid::square(TileGrid::MIN_EDGE).is_some());
    }

    #[test]
    fn a_tile_size_that_would_put_millions_of_tiles_in_range_is_refused() {
        let fine = TileGrid::square(1.0).expect("1px is the minimum, not an error");
        assert_eq!(
            fine.visible_span(rect(0.0, 0.0, 3840.0, 2160.0), 0),
            None,
            "a caller must fall back to untiled buffering rather than allocate \
             eight million layers"
        );
        assert!(
            grid().visible_span(rect(0.0, 0.0, 3840.0, 2160.0), 2).is_some(),
            "the guard must not refuse a 4K viewport at the default tile size"
        );
    }

    #[test]
    fn a_primitive_inside_one_tile_goes_to_that_tile_and_a_spanning_one_to_the_overlay() {
        let grid = grid();
        assert_eq!(
            grid.placement(rect(10.0, 10.0, 40.0, 40.0)),
            TilePlacement::Tile(TileCoord::ORIGIN)
        );
        assert_eq!(
            grid.placement(rect(-100.0, -100.0, 40.0, 40.0)),
            TilePlacement::Tile(TileCoord::new(-1, -1))
        );
        assert_eq!(
            grid.placement(rect(200.0, 10.0, 400.0, 10.0)),
            TilePlacement::Overlay,
            "a wire crossing a tile edge cannot be clipped without a content \
             mask Quad does not have"
        );
        assert_eq!(
            grid.placement(rect(0.0, 0.0, 256.0, 256.0)),
            TilePlacement::Tile(TileCoord::ORIGIN),
            "a primitive filling a tile exactly is inside it, not spanning"
        );
        assert_eq!(grid.placement(Rect::EMPTY), TilePlacement::Overlay);
    }

    #[test]
    fn marking_reports_only_the_tiles_that_were_not_already_resident() {
        let mut residency = TileResidency::new(64);
        let first = residency.mark(
            TileSpan {
                min: TileCoord::ORIGIN,
                max: TileCoord::new(1, 1),
            },
            1,
        );
        assert_eq!(first.len(), 4);
        assert_eq!(
            first,
            vec![
                TileCoord::new(0, 0),
                TileCoord::new(1, 0),
                TileCoord::new(0, 1),
                TileCoord::new(1, 1),
            ]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
        );

        // Pan one column right: only the new column is revealed.
        let second = residency.mark(
            TileSpan {
                min: TileCoord::new(1, 0),
                max: TileCoord::new(2, 1),
            },
            2,
        );
        assert_eq!(
            second,
            vec![TileCoord::new(2, 0), TileCoord::new(2, 1)],
            "§4.3: crossing into a new tile reveals that tile alone"
        );
        assert_eq!(residency.len(), 6);
    }

    #[test]
    fn a_tile_out_of_range_survives_its_interval_and_then_does_not() {
        let mut residency = TileResidency::new(64);
        residency.mark(TileSpan::single(TileCoord::ORIGIN), 1);
        residency.mark(TileSpan::single(TileCoord::new(9, 9)), 2);

        // Visited on frame 1, so it survives an elapsed count of exactly 60 and
        // goes on 61 — the same boundary `Compositor::sweep` draws.
        assert!(residency.sweep(1 + 60, 60).is_empty());
        assert_eq!(residency.len(), 2);
        assert_eq!(
            residency.sweep(2 + 60, 60),
            vec![EvictedTile {
                coord: TileCoord::ORIGIN,
                reason: TileEviction::OutOfRange,
            }],
            "R-N §3.4's interval, with 'unvisited' meaning 'out of range'"
        );
        assert!(residency.contains(TileCoord::new(9, 9)));
    }

    #[test]
    fn the_budget_evicts_the_least_recently_visited_tiles_and_never_a_visible_one() {
        let mut residency = TileResidency::new(4);
        // Eight tiles visited in a known order, one frame each, so the LRU order
        // is unambiguous.
        for index in 0..8i32 {
            residency.mark(TileSpan::single(TileCoord::new(index, 0)), index as u64 + 1);
        }
        assert_eq!(residency.len(), 8);

        // Frame 9 leaves only the last two in range.
        residency.mark(
            TileSpan {
                min: TileCoord::new(6, 0),
                max: TileCoord::new(7, 0),
            },
            9,
        );
        let evicted = residency.sweep(9, 60);
        assert_eq!(residency.len(), 4);
        assert_eq!(
            evicted,
            vec![
                EvictedTile {
                    coord: TileCoord::new(0, 0),
                    reason: TileEviction::Budget
                },
                EvictedTile {
                    coord: TileCoord::new(1, 0),
                    reason: TileEviction::Budget
                },
                EvictedTile {
                    coord: TileCoord::new(2, 0),
                    reason: TileEviction::Budget
                },
                EvictedTile {
                    coord: TileCoord::new(3, 0),
                    reason: TileEviction::Budget
                },
            ],
            "the four oldest go, and the two in range this frame never do"
        );
        assert!(residency.contains(TileCoord::new(6, 0)));
        assert!(residency.contains(TileCoord::new(7, 0)));
        assert_eq!(residency.over_budget(), 0);
    }

    #[test]
    fn a_visible_set_larger_than_the_budget_is_reported_rather_than_thrashed() {
        let mut residency = TileResidency::new(2);
        residency.mark(
            TileSpan {
                min: TileCoord::ORIGIN,
                max: TileCoord::new(2, 1),
            },
            1,
        );
        let evicted = residency.sweep(1, 60);
        assert!(
            evicted.is_empty(),
            "evicting a tile this frame is about to render is not a memory bound"
        );
        assert_eq!(residency.len(), 6);
        assert_eq!(residency.over_budget(), 4);
    }

    /// §9's risk table's exact shape: "fast diagonal movement, frequent
    /// direction reversal" keeping many tiles inside the eviction interval at
    /// once, which the per-tile timer alone does not bound.
    ///
    /// The claim is comparative on purpose. A test that only checked "the
    /// resident count stayed under the budget" would pass just as well against a
    /// pan that never had many tiles live in the first place, which would make
    /// the budget untested rather than proven. So the same walk runs twice —
    /// once with a budget, once with one large enough to be inert — and the
    /// timer-only arm is asserted to blow past what the bounded arm holds.
    #[test]
    fn the_resident_tile_budget_is_what_bounds_an_erratic_pan_not_the_timer() {
        let grid = grid();
        let walk = |budget: usize| {
            let mut residency = TileResidency::new(budget);
            let mut peak = 0usize;
            for frame in 0..400u64 {
                let phase = frame as f32 * 0.37;
                let viewport = rect(phase.sin() * 4_000.0, phase.cos() * 4_000.0, 900.0, 600.0);
                let span = grid
                    .visible_span(viewport, 1)
                    .expect("a 900x600 viewport is usable at 256px");
                residency.mark(span, frame);
                residency.sweep(frame, 60);
                peak = peak.max(residency.len());
            }
            (peak, residency.over_budget())
        };

        let (bounded_peak, over) = walk(96);
        let (timer_only_peak, _) = walk(1_000_000);

        assert!(
            bounded_peak <= 96,
            "the budget did not bound an erratic pan: peak {bounded_peak}"
        );
        assert_eq!(
            over, 0,
            "96 is above this viewport's in-range set, so nothing should be \
             left unaccounted"
        );
        assert!(
            timer_only_peak > bounded_peak * 2,
            "the timer-only arm peaked at {timer_only_peak} against the bounded \
             arm's {bounded_peak}, so this walk does not exercise what the \
             budget is for"
        );
    }

    /// The other half of the same story, and the honest one: a budget *below*
    /// the in-range set cannot be met, and the mechanism says so rather than
    /// evicting tiles the same frame is about to render.
    #[test]
    fn a_budget_under_the_viewports_own_tile_count_reports_itself_unmet() {
        let grid = grid();
        let mut residency = TileResidency::new(8);
        let viewport = rect(0.0, 0.0, 900.0, 600.0);
        let span = grid
            .visible_span(viewport, 1)
            .expect("a 900x600 viewport is usable at 256px");
        let visible = span.tile_count() as usize;
        assert!(visible > 8, "the premise: this viewport needs more than 8 tiles");

        residency.mark(span, 1);
        residency.sweep(1, 60);
        assert_eq!(residency.len(), visible);
        assert_eq!(
            residency.over_budget(),
            visible - 8,
            "the shortfall is reported, not absorbed"
        );
    }

    #[test]
    fn tile_descriptors_encode_as_one_padded_vec4_each_with_signed_coordinates() {
        let mut bytes = Vec::new();
        encode_tiles(
            &[TileDescriptor {
                coord: TileCoord::new(-3, 7),
                base: 64,
                count: 12,
            }],
            &mut bytes,
        );
        assert_eq!(bytes.len(), TILE_DESCRIPTOR_STRIDE);
        assert_eq!(&bytes[0..4], &(-3i32).to_le_bytes());
        assert_eq!(&bytes[4..8], &7i32.to_le_bytes());
        assert_eq!(&bytes[8..12], &64u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &12u32.to_le_bytes());
    }

    #[test]
    fn an_out_of_range_tile_gets_a_zero_count_slot_and_keeps_its_base() {
        let grid = grid();
        let tiles = [
            TileDescriptor {
                coord: TileCoord::ORIGIN,
                base: 0,
                count: 10,
            },
            TileDescriptor {
                coord: TileCoord::new(40, 0),
                base: 10,
                count: 7,
            },
        ];
        let visibility = tile_visibility(&grid, &tiles, rect(0.0, 0.0, 256.0, 256.0), 0);
        assert_eq!(visibility.slots, vec![[0, 10, 0, 0], [10, 0, 0, 0]]);
        assert_eq!(visibility.in_range, vec![1, 0]);
        assert_eq!(visibility.visible_count(), 1);
    }

    #[test]
    fn the_slot_encoding_is_byte_identical_to_the_indirect_pass_own() {
        use crate::indirect::{DrawSlot, encode_slots};
        use crate::patch::primitive::PrimitiveKind;
        use crate::scene::layer::LayerId;

        let grid = grid();
        let tiles = [
            TileDescriptor {
                coord: TileCoord::ORIGIN,
                base: 0,
                count: 10,
            },
            TileDescriptor {
                coord: TileCoord::new(40, 0),
                base: 10,
                count: 7,
            },
        ];
        let visibility = tile_visibility(&grid, &tiles, rect(0.0, 0.0, 256.0, 256.0), 0);
        let mut ours = Vec::new();
        visibility.encode_slots(&mut ours);

        // The same records as the slot table Phase 4 already encodes, so the
        // tile pass genuinely feeds the existing mechanism rather than a
        // parallel one that happens to look similar.
        let mut theirs = Vec::new();
        encode_slots(
            &[
                DrawSlot {
                    layer: LayerId::from_raw(1),
                    kind: PrimitiveKind::Quad,
                    base: 0,
                    count: 10,
                },
                DrawSlot {
                    layer: LayerId::from_raw(2),
                    kind: PrimitiveKind::Quad,
                    base: 10,
                    count: 0,
                },
            ],
            &mut theirs,
        );
        assert_eq!(ours, theirs);
    }
}
