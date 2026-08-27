//! A scripted UI walk over a realistic editor-shaped scene, driven through the
//! real patch protocol into a real [`Scene`].
//! See docs/gpu-native-architecture.md §5.2, §8 Phase 3; R-N §8.5.
//!
//! # What "realistic" is doing here
//!
//! Phase 0's Spike A used 100,000 quads in 200 spatially disjoint clusters, and
//! its own write-up flags what that bought it: the clusters bounded every
//! neighbour search for free, and "a real generic scene would need real spatial
//! partitioning." §8's Phase 3 gate answers that by asking for a scene built
//! through the real APIs rather than a synthetic buffer. So the scene here is
//! shaped like the application this crate exists for — window chrome, a
//! scrolling tree panel, a node-graph viewport, a docked inspector, and a modal
//! — with the properties that actually make occlusion interesting:
//!
//! - **Real occluders**: the inspector panel and the modal dialog are opaque
//!   and painted last, so they genuinely hide content beneath them.
//! - **Real non-occluders**: node drop shadows and the modal scrim are
//!   translucent, node bodies are rounded and bordered, and label bars are
//!   translucent. Every one of R-N §8.3's rejection reasons appears somewhere.
//! - **Real overlap in paint order**: nodes are placed on a jittered grid so
//!   some overlap and step each other's painter order, and the panels overlap
//!   the viewport.
//! - **No spatial partitioning handed to the algorithm.** Nothing about the
//!   layout is communicated to either compute pass; both see a flat list.
//!
//! # And what it deliberately is not
//!
//! Not `wgpui-widgets` elements, and not a `Description`/`FramePlan`. §8 puts
//! the element vocabulary elsewhere and Phase 2 already proved the emission
//! arrow; what Phase 3's gate needs is a *scene*, resident in real slabs,
//! reachable through `PrimitiveStore`, whose primitives the compute passes then
//! consume. Going through `wgpui-widgets` would add a layer neither gate asks
//! about between the generator and the thing under test.

use crate::boundary::compositor::Compositor;
use crate::boundary::policy::{BoundaryPolicy, Buffering, Size};
use crate::geometry::Rect;
use crate::invalidation::axes::Invalidation;
use crate::occlusion::{CoverageItem, PoisonRegion, quad_coverage_item};
use crate::patch::apply::{ScenePatch, UploadPlan, apply};
use crate::patch::primitive::Quad;
use crate::patch::{PatchError, RecordKey};
use crate::scene::layer::{Layer, LayerTransform};
use crate::scene::{BoundaryId, LayerId, LayerKey, Scene, TileCoord, TileGrid};
use std::collections::HashMap;

/// What one frame of the walk looks like.
#[derive(Clone, Debug, PartialEq)]
pub struct UiSceneSpec {
    /// Window width in pixels.
    pub width: f32,
    /// Window height in pixels.
    pub height: f32,
    /// Rows in the left tree panel. The panel's content is taller than the
    /// window, exactly as a real list's is.
    pub list_rows: u32,
    /// Nodes in the centre viewport.
    pub nodes: u32,
    /// How far the tree panel is scrolled, in pixels.
    pub list_scroll: f32,
    /// Which row is highlighted.
    pub selected_row: u32,
    /// Whether the right-hand inspector is docked open.
    pub inspector_open: bool,
    /// Whether a modal dialog is up.
    pub modal_open: bool,
    /// Uniform scale on row height and node size, i.e. how zoomed-out the
    /// content is.
    ///
    /// At `1.0` the scene is a normal editor and a large `list_rows`/`nodes`
    /// puts most of the content *below* the window — which is a real and
    /// important shape (a retained layer holds its whole content, §5.0), but
    /// one where most primitives clip away to nothing and the occlusion pass
    /// early-outs on them. Scaling down packs the same primitive count into the
    /// visible area instead, which is the shape where occlusion has real work
    /// to do. The benchmark measures both, because either alone would be a
    /// partial answer.
    pub content_scale: f32,
}

impl UiSceneSpec {
    /// A small scene, sized for the differential harness: enough structure to
    /// exercise every rule, small enough to rasterize every frame twice.
    pub fn small() -> UiSceneSpec {
        UiSceneSpec {
            width: 480.0,
            height: 320.0,
            list_rows: 40,
            nodes: 24,
            list_scroll: 0.0,
            selected_row: 3,
            inspector_open: true,
            modal_open: false,
            content_scale: 1.0,
        }
    }

    /// A large scene at normal zoom, sized for the performance gate.
    ///
    /// Most of its content sits below the window: a retained layer holds its
    /// whole list and its whole graph, not only the visible part. See
    /// [`UiSceneSpec::content_scale`] for the other half of that story.
    pub fn large(list_rows: u32, nodes: u32) -> UiSceneSpec {
        UiSceneSpec {
            width: 2560.0,
            height: 1440.0,
            list_rows,
            nodes,
            list_scroll: 0.0,
            selected_row: 17,
            inspector_open: true,
            modal_open: true,
            content_scale: 1.0,
        }
    }

    /// The same scene zoomed out until the content fills the window, so the
    /// primitives the passes see are overwhelmingly *visible* ones.
    pub fn large_dense(list_rows: u32, nodes: u32, content_scale: f32) -> UiSceneSpec {
        UiSceneSpec {
            content_scale,
            ..UiSceneSpec::large(list_rows, nodes)
        }
    }
}

/// One frame's primitives and the filter regions that read through them.
#[derive(Clone, Debug)]
pub struct UiFrame {
    /// What this frame is doing, for a failing assertion to name.
    pub label: String,
    /// Primitives in paint order.
    pub quads: Vec<Quad>,
    /// Backdrop-filter / filter-group regions, already dilated by their blur
    /// radius (R-N §8.3's last two conditions).
    pub poison: Vec<PoisonRegion>,
    /// The window rectangle, which is also every primitive's clip here.
    pub clip: Rect,
}

impl UiFrame {
    /// This frame's primitives as occlusion inputs.
    pub fn coverage_items(&self) -> Vec<CoverageItem> {
        self.quads
            .iter()
            .map(|quad| quad_coverage_item(quad, self.clip, false))
            .collect()
    }

    /// This frame's primitives as ordering inputs.
    pub fn bounds(&self) -> Vec<Rect> {
        self.quads
            .iter()
            .map(|quad| Rect::from_origin_size(quad.origin, quad.size))
            .collect()
    }
}

/// An upper bound on how many primitives a spec produces, for sizing a
/// benchmark.
///
/// A bound rather than a count: rows scrolled off the top of the panel are not
/// emitted (a real list virtualises them), and the inspector stops laying out
/// fields once it runs out of height. Both make the real number smaller.
pub fn quad_count_estimate(spec: &UiSceneSpec) -> u32 {
    const CHROME: u32 = 5;
    const PER_ROW: u32 = 3;
    const PER_NODE: u32 = 6;
    const INSPECTOR: u32 = 1 + 2 * INSPECTOR_FIELDS;
    const MODAL: u32 = 8;
    CHROME
        + spec.list_rows * PER_ROW
        + spec.nodes * PER_NODE
        + if spec.inspector_open { INSPECTOR } else { 0 }
        + if spec.modal_open { MODAL } else { 0 }
}

const TITLE_BAR_HEIGHT: f32 = 32.0;
const STATUS_BAR_HEIGHT: f32 = 24.0;
const ROW_HEIGHT: f32 = 22.0;
const NODE_WIDTH: f32 = 120.0;
const NODE_HEIGHT: f32 = 64.0;
/// Column pitch as a fraction of a node's width. Below one, so horizontal
/// neighbours overlap and genuinely step each other's painter order.
const NODE_CELL_FRACTION: f32 = 0.55;
/// Row pitch as a fraction of a node's height. Above one, so vertical
/// neighbours only touch when the jitter pushes them together.
const NODE_ROW_PITCH_FRACTION: f32 = 1.45;
const INSPECTOR_FIELDS: u32 = 20;

fn solid(red: f32, green: f32, blue: f32) -> [f32; 4] {
    [red, green, blue, 1.0]
}

fn translucent(red: f32, green: f32, blue: f32, alpha: f32) -> [f32; 4] {
    [red, green, blue, alpha]
}

fn plain(bounds: Rect, background: [f32; 4]) -> Quad {
    Quad {
        origin: [bounds.min_x, bounds.min_y],
        size: [bounds.width(), bounds.height()],
        background,
        border_color: [0.0, 0.0, 0.0, 0.0],
        corner_radius: 0.0,
        border_width: 0.0,
    }
}

fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
    Rect::from_origin_size([x, y], [width, height])
}

/// Deterministic jitter, so a failing frame is reproducible from its index.
fn jitter(seed: u32, span: f32) -> f32 {
    let mut value = seed.wrapping_mul(0x9E37_79B9);
    value ^= value >> 15;
    value = value.wrapping_mul(0x85EB_CA6B);
    value ^= value >> 13;
    ((value % 1000) as f32 / 1000.0 - 0.5) * span
}

/// Build one frame's primitives, in paint order.
pub fn build_frame(label: &str, spec: &UiSceneSpec) -> UiFrame {
    let mut quads = Vec::with_capacity(quad_count_estimate(spec) as usize);
    let clip = rect(0.0, 0.0, spec.width, spec.height);

    // --- The window's floor.
    quads.push(plain(clip, solid(0.11, 0.11, 0.13)));

    let panel_width = (spec.width * 0.22).floor();
    let inspector_width = if spec.inspector_open {
        (spec.width * 0.24).floor()
    } else {
        0.0
    };
    let content_top = TITLE_BAR_HEIGHT;
    let content_bottom = spec.height - STATUS_BAR_HEIGHT;

    // --- Left tree panel: an opaque background with a taller-than-window list
    // of rows scrolled over it.
    quads.push(plain(
        rect(
            0.0,
            content_top,
            panel_width,
            content_bottom - content_top,
        ),
        solid(0.13, 0.13, 0.16),
    ));
    let scale = spec.content_scale.clamp(0.02, 4.0);
    let row_height = (ROW_HEIGHT * scale).max(1.0);
    for row in 0..spec.list_rows {
        let top = content_top + row as f32 * row_height - spec.list_scroll;
        // Rows scrolled past the panel's top edge are not emitted, which is
        // what a real virtualised list does. Rows below the window's bottom
        // *are* emitted: a list's slab holds its whole content, and that is the
        // residency §5.0 cares about.
        if top + row_height <= content_top {
            continue;
        }
        let selected = row == spec.selected_row;
        let background = if selected {
            solid(0.22, 0.34, 0.52)
        } else if row % 2 == 0 {
            solid(0.15, 0.15, 0.18)
        } else {
            translucent(0.20, 0.20, 0.24, 0.35)
        };
        quads.push(plain(rect(0.0, top, panel_width, row_height), background));
        // An icon: opaque but rounded, so its opaque region is inset.
        let icon = (14.0 * scale).max(1.0);
        quads.push(Quad {
            origin: [6.0 * scale, top + (row_height - icon) * 0.5],
            size: [icon, icon],
            background: solid(0.55, 0.62, 0.35),
            border_color: [0.0, 0.0, 0.0, 0.0],
            corner_radius: (4.0 * scale).max(0.0),
            border_width: 0.0,
        });
        // A label bar standing in for shaped text: translucent, never occludes.
        let label_left = 26.0 * scale;
        quads.push(plain(
            rect(
                label_left,
                top + row_height * 0.35,
                (panel_width - label_left - 8.0 * scale).max(1.0),
                (8.0 * scale).max(1.0),
            ),
            translucent(0.85, 0.86, 0.90, 0.75),
        ));
    }

    // --- Centre viewport. It spans the whole content width: the inspector is
    // an overlay docked *over* the canvas rather than a column reserved beside
    // it, which is both what a real editor does and what makes the nodes
    // underneath it genuinely occluded rather than merely adjacent.
    let viewport = rect(
        panel_width,
        content_top,
        spec.width - panel_width,
        content_bottom - content_top,
    );
    quads.push(plain(viewport, solid(0.09, 0.09, 0.11)));

    // Nodes are placed on a cell grid narrower than a node is wide, so
    // neighbours overlap and genuinely step each other's painter order. The
    // grid runs the full viewport width — including under the inspector — and
    // extends downward past the window, so the layer's residency is the whole
    // graph rather than only its visible part.
    let node_width = (NODE_WIDTH * scale).max(2.0);
    let node_height = (NODE_HEIGHT * scale).max(2.0);
    let cell_width = (node_width * NODE_CELL_FRACTION).max(1.0);
    let row_pitch = (node_height * NODE_ROW_PITCH_FRACTION).max(1.0);
    let columns = ((viewport.width() / cell_width) as u32).max(1);
    for node in 0..spec.nodes {
        let column = node % columns;
        let row = node / columns;
        let x = viewport.min_x
            + column as f32 * cell_width
            + jitter(node.wrapping_mul(3) + 1, cell_width * 0.3);
        let y = viewport.min_y + 16.0 * scale + row as f32 * row_pitch
            + jitter(node.wrapping_mul(7) + 2, row_pitch * 0.2);
        // Drop shadow: translucent, so it can be culled but never occludes.
        quads.push(plain(
            rect(
                x + 3.0 * scale,
                y + 3.0 * scale,
                node_width,
                node_height,
            ),
            translucent(0.0, 0.0, 0.0, 0.35),
        ));
        // Body: opaque, rounded, with a translucent hairline border — so its
        // opaque region is inset by the larger of the two.
        quads.push(Quad {
            origin: [x, y],
            size: [node_width, node_height],
            background: solid(0.19, 0.20, 0.24),
            border_color: translucent(0.55, 0.58, 0.66, 0.5),
            corner_radius: 6.0 * scale,
            border_width: 1.0 * scale,
        });
        quads.push(plain(
            rect(
                x + node_width * 0.05,
                y + node_height * 0.09,
                node_width * 0.9,
                (node_height * 0.22).max(1.0),
            ),
            solid(0.28, 0.32, 0.42),
        ));
        let port = (8.0 * scale).max(1.0);
        for index in 0..3u32 {
            quads.push(Quad {
                origin: [
                    x - port * 0.5,
                    y + node_height * 0.4 + index as f32 * port * 1.5,
                ],
                size: [port, port],
                background: solid(0.72, 0.66, 0.30),
                border_color: [0.0, 0.0, 0.0, 0.0],
                corner_radius: port * 0.5,
                border_width: 0.0,
            });
        }
    }

    // --- Docked inspector, painted after the viewport so it genuinely covers
    // whatever nodes reach under it. This is the scene's main occluder.
    if spec.inspector_open {
        let inspector = rect(
            spec.width - inspector_width,
            content_top,
            inspector_width,
            content_bottom - content_top,
        );
        quads.push(plain(inspector, solid(0.13, 0.13, 0.16)));
        for field in 0..INSPECTOR_FIELDS {
            let top = inspector.min_y + 8.0 + field as f32 * 26.0;
            if top + 22.0 > inspector.max_y {
                break;
            }
            quads.push(plain(
                rect(inspector.min_x + 8.0, top, inspector_width - 16.0, 22.0),
                translucent(0.20, 0.20, 0.24, 0.6),
            ));
            quads.push(plain(
                rect(
                    inspector.min_x + inspector_width * 0.5,
                    top + 3.0,
                    inspector_width * 0.5 - 12.0,
                    16.0,
                ),
                solid(0.10, 0.10, 0.12),
            ));
        }
    }

    // --- Modal: a translucent scrim that never occludes over an opaque dialog
    // that very much does.
    if spec.modal_open {
        quads.push(plain(clip, translucent(0.0, 0.0, 0.0, 0.45)));
        // Sized as a fraction of the window rather than in absolute pixels, so
        // the small harness scene and the large benchmark scene have the same
        // shape and the dialog never swallows the whole window.
        let dialog_width = (spec.width * 0.4).floor();
        let dialog_height = (spec.height * 0.4).floor();
        let dialog = rect(
            ((spec.width - dialog_width) * 0.5).floor(),
            ((spec.height - dialog_height) * 0.5).floor(),
            dialog_width,
            dialog_height,
        );
        quads.push(Quad {
            origin: [dialog.min_x, dialog.min_y],
            size: [dialog.width(), dialog.height()],
            background: solid(0.17, 0.17, 0.21),
            border_color: [0.0, 0.0, 0.0, 0.0],
            corner_radius: 8.0,
            border_width: 0.0,
        });
        quads.push(plain(
            rect(dialog.min_x, dialog.min_y + 40.0, dialog.width(), 1.0),
            translucent(1.0, 1.0, 1.0, 0.12),
        ));
        for line in 0..4u32 {
            quads.push(plain(
                rect(
                    dialog.min_x + 20.0,
                    dialog.min_y + 60.0 + line as f32 * 18.0,
                    dialog.width() - 40.0,
                    9.0,
                ),
                translucent(0.85, 0.86, 0.90, 0.7),
            ));
        }
        quads.push(plain(
            rect(dialog.max_x - 110.0, dialog.max_y - 44.0, 90.0, 28.0),
            solid(0.26, 0.40, 0.62),
        ));
    }

    // --- Window chrome, painted last.
    //
    // Real chrome overlays content and the content beneath it is clipped away;
    // `Quad` has no per-primitive content mask yet (`docs/phase-1-results.md`
    // §2), so painting the bars over the content is the honest model of that,
    // and it is what gives the scene an occluder that is present in *every*
    // frame — including the one where both the inspector and the modal are
    // closed. A harness where some frame culls nothing tests nothing on that
    // frame.
    quads.push(plain(
        rect(0.0, 0.0, spec.width, TITLE_BAR_HEIGHT),
        solid(0.16, 0.16, 0.19),
    ));
    quads.push(plain(
        rect(
            0.0,
            spec.height - STATUS_BAR_HEIGHT,
            spec.width,
            STATUS_BAR_HEIGHT,
        ),
        solid(0.14, 0.14, 0.17),
    ));

    // --- One backdrop-filter region: a small frosted breadcrumb chip at the
    // top of the tree panel, poisoning everything beneath it. Nothing
    // rasterizes a blur here; what it exercises is R-N §8.3's poisoning rule
    // and its blur margin, and since poisoning only ever *prevents* culls, the
    // differential harness stays sound without a blur in the rasterizer.
    //
    // Its placement is not incidental. R-N's rule is that *any* intersection
    // poisons the whole primitive, so a filter drawn across a panel edge would
    // poison that whole panel — and a filter over the docked inspector or the
    // modal dialog would take every cull in the scene with it, leaving the
    // harness measuring nothing. `the_filter_chip_does_not_poison_the_scenes_real_occluders`
    // asserts that rather than trusting this paragraph.
    let poison = vec![PoisonRegion {
        region: rect(8.0, TITLE_BAR_HEIGHT + 8.0, 96.0, 14.0).dilate(6.0),
        above_index: u32::try_from(quads.len()).unwrap_or(u32::MAX),
    }];

    UiFrame {
        label: label.to_string(),
        quads,
        poison,
        clip,
    }
}

/// A scripted walk: the frames R-N §8.5 asks CI to run validate mode over.
///
/// Each step changes one thing, the way a user does — scroll, select, open the
/// modal, close the inspector — so a divergence names a transition rather than
/// a snapshot.
pub fn scripted_walk(base: &UiSceneSpec) -> Vec<UiFrame> {
    let mut frames = Vec::new();
    let mut push = |label: &str, spec: &UiSceneSpec| frames.push(build_frame(label, spec));

    let mut spec = base.clone();
    push("initial", &spec);

    for step in 1..=4u32 {
        spec.list_scroll = step as f32 * ROW_HEIGHT * base.content_scale * 1.5;
        push(&format!("scrolled {step}"), &spec);
    }

    spec.selected_row = spec.selected_row.wrapping_add(9) % base.list_rows.max(1);
    push("selection moved", &spec);

    spec.modal_open = !spec.modal_open;
    push("modal toggled", &spec);

    spec.list_scroll += ROW_HEIGHT * base.content_scale * 3.0;
    push("scrolled under the modal", &spec);

    spec.modal_open = !spec.modal_open;
    push("modal toggled back", &spec);

    spec.inspector_open = !spec.inspector_open;
    push("inspector toggled", &spec);

    spec.nodes = spec.nodes.saturating_sub(5);
    push("nodes removed", &spec);

    spec.nodes += 9;
    push("nodes added", &spec);

    spec.inspector_open = !spec.inspector_open;
    spec.list_scroll = 0.0;
    push("returned to the start", &spec);

    frames
}

/// Drives a sequence of frames into one real [`Scene`], through
/// [`ScenePatch`] — inserts, in-place updates, and removals, exactly as an
/// emitter would.
pub struct SceneDriver {
    /// The scene under test.
    pub scene: Scene,
    /// The single layer these frames live in.
    pub layer: LayerId,
    resident: usize,
}

impl SceneDriver {
    /// A driver over an empty scene with one declared layer.
    pub fn new() -> SceneDriver {
        let mut scene = Scene::new();
        let layer = scene.layer(LayerKey::untiled(BoundaryId::ROOT));
        SceneDriver {
            scene,
            layer,
            resident: 0,
        }
    }

    /// Bring the scene's layer to exactly `quads`, and return what that cost to
    /// upload.
    ///
    /// Records keep their index across frames, so an unchanged primitive
    /// produces no patch at all and a changed one produces §5.0's O(1) in-place
    /// update — the same cross-frame addressing Phase 2's emitter established,
    /// applied to a generator rather than to an element tree.
    pub fn apply_frame(&mut self, quads: &[Quad]) -> Result<UploadPlan, PatchError> {
        let mut patch = ScenePatch::new();
        let shared = self.resident.min(quads.len());
        for index in 0..shared {
            let key = RecordKey::from_raw(index as u64 + 1);
            let Some(quad) = quads.get(index) else {
                continue;
            };
            if self.scene.quads.get(self.layer, key) != Some(quad) {
                patch.quads.update(self.layer, key, *quad);
            }
        }
        for index in shared..quads.len() {
            let Some(quad) = quads.get(index) else {
                continue;
            };
            patch.quads.append(
                self.layer,
                RecordKey::from_raw(index as u64 + 1),
                u32::try_from(index).unwrap_or(u32::MAX),
                *quad,
            );
        }
        for index in (quads.len()..self.resident).rev() {
            patch
                .quads
                .remove(self.layer, RecordKey::from_raw(index as u64 + 1));
        }
        self.resident = quads.len();
        apply(&mut self.scene, &patch)
    }

    /// The layer's primitives, read back out of the store in paint order.
    ///
    /// Going through `keys`/`get` rather than keeping the input `Vec` is the
    /// point: it is what makes the compute passes' input come from the real
    /// scene rather than from the generator that seeded it.
    pub fn resident_quads(&self) -> Vec<Quad> {
        self.scene
            .quads
            .keys(self.layer)
            .into_iter()
            .filter_map(|key| self.scene.quads.get(self.layer, key).copied())
            .collect()
    }
}

impl Default for SceneDriver {
    fn default() -> Self {
        Self::new()
    }
}

/// The same walk driven into *several* layers instead of one — what Phase 4
/// needs, because a fixed draw sequence "one per (layer, kind) slot" is
/// untestable against a scene with one slot in it.
///
/// A frame's quads are split into contiguous chunks in paint order, one chunk
/// per layer. Contiguous rather than by panel, deliberately: paint order across
/// layers *is* layer order, in the legacy backend and here, so a contiguous
/// split is the faithful shape and it needs nothing from
/// [`build_frame`]'s internal structure. What it does not model is a layer whose
/// content interleaves another's in paint order, which the layer concept does
/// not allow anyway.
pub struct MultiLayerSceneDriver {
    /// The scene under test.
    pub scene: Scene,
    /// The window rectangle every primitive clips to.
    pub clip: Rect,
    /// The frame's filter regions.
    pub poison: Vec<PoisonRegion>,
    layers: Vec<LayerId>,
    resident: Vec<usize>,
}

impl MultiLayerSceneDriver {
    /// A driver over an empty scene with `layer_count` declared layers.
    pub fn new(layer_count: usize) -> MultiLayerSceneDriver {
        let mut scene = Scene::new();
        let layers: Vec<LayerId> = (0..layer_count.max(1))
            .map(|index| {
                scene.layer(LayerKey::untiled(BoundaryId::from_raw(index as u64 + 1)))
            })
            .collect();
        let resident = vec![0usize; layers.len()];
        MultiLayerSceneDriver {
            scene,
            clip: Rect::EMPTY,
            poison: Vec::new(),
            layers,
            resident,
        }
    }

    /// Every declared layer, in draw order.
    pub fn layers(&self) -> &[LayerId] {
        &self.layers
    }

    /// Bring the scene to `frame`, splitting its quads across the layers.
    pub fn apply_frame(&mut self, frame: &UiFrame) -> Result<(), PatchError> {
        self.clip = frame.clip;
        self.poison.clone_from(&frame.poison);
        let layer_count = self.layers.len();
        let chunk = frame.quads.len().div_ceil(layer_count.max(1)).max(1);
        for index in 0..layer_count {
            let start = (index * chunk).min(frame.quads.len());
            let end = ((index + 1) * chunk).min(frame.quads.len());
            let quads = frame.quads.get(start..end).unwrap_or(&[]).to_vec();
            self.set_layer(index, &quads)?;
        }
        Ok(())
    }

    /// Bring one layer to exactly `quads`, through the real patch protocol.
    pub fn set_layer(&mut self, index: usize, quads: &[Quad]) -> Result<(), PatchError> {
        let Some(layer) = self.layers.get(index).copied() else {
            return Ok(());
        };
        let resident = self.resident.get(index).copied().unwrap_or(0);
        let mut patch = ScenePatch::new();
        let shared = resident.min(quads.len());
        for position in 0..shared {
            let key = RecordKey::from_raw(position as u64 + 1);
            let Some(quad) = quads.get(position) else {
                continue;
            };
            if self.scene.quads.get(layer, key) != Some(quad) {
                patch.quads.update(layer, key, *quad);
            }
        }
        for position in shared..quads.len() {
            let Some(quad) = quads.get(position) else {
                continue;
            };
            patch.quads.append(
                layer,
                RecordKey::from_raw(position as u64 + 1),
                u32::try_from(position).unwrap_or(u32::MAX),
                *quad,
            );
        }
        for position in (quads.len()..resident).rev() {
            patch
                .quads
                .remove(layer, RecordKey::from_raw(position as u64 + 1));
        }
        if let Some(slot) = self.resident.get_mut(index) {
            *slot = quads.len();
        }
        apply(&mut self.scene, &patch)?;
        Ok(())
    }

    /// One layer's primitives, read back out of the store in paint order.
    pub fn layer_quads(&self, layer: LayerId) -> Vec<Quad> {
        self.scene
            .quads
            .keys(layer)
            .into_iter()
            .filter_map(|key| self.scene.quads.get(layer, key).copied())
            .collect()
    }

    /// One layer's primitives as occlusion inputs.
    pub fn coverage_items(&self, layer: LayerId) -> Vec<CoverageItem> {
        self.layer_quads(layer)
            .iter()
            .map(|quad| quad_coverage_item(quad, self.clip, false))
            .collect()
    }
}

/// A node-graph canvas on an unbounded content plane, driven through
/// `Buffering::Tiled` into a real [`Scene`].
/// See docs/gpu-native-architecture.md §4.3, §8 Phase 4.5.
///
/// # Why this is a second generator rather than a `UiSceneSpec` flag
///
/// [`build_frame`]'s scene is an *editor*: everything in it is positioned
/// relative to a window, and its viewport is one panel among several. §4.3's
/// case is the opposite shape — content at arbitrary positions on a plane with
/// no origin corner, a window that moves over it, and no linear index to
/// virtualize by. Bolting a pan offset onto the editor spec would produce a
/// scene whose content still fundamentally lives in window space, which is the
/// exact thing `Buffering::Margin` already handles and `Tiled` exists because it
/// does not.
///
/// # What it measures, and why it measures it this way
///
/// Every frame emits patches for **every visible tile**, not only the revealed
/// ones. That is deliberate and it is what makes the gate a measurement rather
/// than a restatement: if the driver skipped resident tiles, "panning costs zero
/// render work" would be true because the harness declined to do any. Instead
/// the harness offers the work every frame and the patch protocol's own
/// cross-frame addressing (§5.0) is what reduces it to nothing — so a
/// [`TiledFrameStats`] reading zero is a fact about the mechanism.
pub struct TiledCanvasDriver {
    /// The scene under test.
    pub scene: Scene,
    /// The compositor holding this boundary's tile residency.
    pub compositor: Compositor,
    /// The boundary the canvas lives in.
    pub boundary: BoundaryId,
    /// The policy it was declared with.
    pub policy: BoundaryPolicy,
    /// The window rectangle the canvas is seen through.
    pub viewport: Rect,
    graph: NodeGraph,
    resident: HashMap<LayerId, usize>,
    frame: u64,
}

/// What one frame of a tiled pan cost, in the terms §8's Phase 4.5 gate is
/// written in.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TiledFrameStats {
    /// Tiles newly in range this frame — the ones that must render.
    pub revealed: Vec<TileCoord>,
    /// Tiles in range this frame.
    pub visible_tiles: usize,
    /// Tiles resident after the sweep.
    pub resident_tiles: usize,
    /// Tiles evicted this frame.
    pub evicted: usize,
    /// Resident tiles the budget could not account for.
    pub over_budget: usize,
    /// Layers whose composite transform this frame changed — **one per visible
    /// tile on a pan**, which is the gate's own wording.
    pub transform_updates: usize,
    /// Layers this frame created, i.e. tiles that became layers.
    pub layers_created: usize,
    /// Layers this frame destroyed.
    pub layers_removed: usize,
    /// Primitives written into the scene this frame — inserts plus updates.
    /// **The render/reconcile work the gate requires to be zero on a pan inside
    /// the resident grid.**
    pub primitives_written: usize,
    /// How many of [`TiledFrameStats::primitives_written`] landed on the
    /// unbuffered overlay rather than in a tile.
    ///
    /// Tracked separately because of what it turned out to be worth: a tile
    /// crossing dirties the overlay as well as the revealed tiles, since a wire
    /// reaching into the new column is spanning content and spanning content
    /// lives there. That is a real consequence of the rule
    /// [`crate::scene::TilePlacement`] picks, and separating the two counts is
    /// what lets the gate state it instead of averaging it away.
    pub overlay_primitives_written: usize,
    /// Bytes §5.0's upload instructions cover this frame.
    pub upload_bytes: u64,
    /// Layers that ended the frame carrying `DISPLAY`.
    pub display_layers: Vec<LayerId>,
    /// Layers that ended the frame carrying `TRANSFORM` and nothing else.
    pub transform_only_layers: usize,
}

impl TiledFrameStats {
    /// Whether this frame did no rendering, reconciling, or uploading at all.
    pub fn is_transform_only(&self) -> bool {
        self.primitives_written == 0
            && self.upload_bytes == 0
            && self.display_layers.is_empty()
            && self.layers_created == 0
    }
}

/// The graph's content, laid out once on the plane and never re-laid-out.
struct NodeGraph {
    quads: Vec<(Rect, Quad)>,
}

/// How big a node-graph canvas is and what is on it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NodeGraphSpec {
    /// How many node columns the plane holds.
    pub columns: u32,
    /// How many node rows.
    pub rows: u32,
    /// Horizontal distance between node origins.
    pub column_pitch: f32,
    /// Vertical distance between node origins.
    pub row_pitch: f32,
    /// Whether wires between horizontally adjacent nodes are emitted. Wires are
    /// the content that genuinely spans tiles, so turning them off is what
    /// isolates the overlay's cost.
    pub wires: bool,
}

impl NodeGraphSpec {
    /// A plane large enough that no realistic pan ever sees all of it.
    pub fn large() -> NodeGraphSpec {
        NodeGraphSpec {
            columns: 40,
            rows: 40,
            column_pitch: 220.0,
            row_pitch: 180.0,
            wires: true,
        }
    }
}

const GRAPH_NODE_WIDTH: f32 = 130.0;
const GRAPH_NODE_HEIGHT: f32 = 70.0;

impl NodeGraph {
    fn build(spec: &NodeGraphSpec) -> NodeGraph {
        let mut quads = Vec::new();
        let mut push = |bounds: Rect, quad: Quad| quads.push((bounds, quad));
        let origin_of = |column: u32, row: u32| {
            // Centred on the plane's origin, so panning in every direction
            // reaches content and negative tile coordinates are ordinary rather
            // than an edge case.
            let x = (column as f32 - spec.columns as f32 * 0.5) * spec.column_pitch
                + jitter(column.wrapping_mul(31).wrapping_add(row), spec.column_pitch * 0.25);
            let y = (row as f32 - spec.rows as f32 * 0.5) * spec.row_pitch
                + jitter(row.wrapping_mul(17).wrapping_add(column), spec.row_pitch * 0.2);
            [x, y]
        };

        for row in 0..spec.rows {
            for column in 0..spec.columns {
                let [x, y] = origin_of(column, row);
                let body = rect(x, y, GRAPH_NODE_WIDTH, GRAPH_NODE_HEIGHT);
                push(
                    body,
                    Quad {
                        origin: [body.min_x, body.min_y],
                        size: [body.width(), body.height()],
                        background: solid(0.19, 0.20, 0.24),
                        border_color: translucent(0.55, 0.58, 0.66, 0.5),
                        corner_radius: 6.0,
                        border_width: 1.0,
                    },
                );
                let header = rect(x + 6.0, y + 6.0, GRAPH_NODE_WIDTH - 12.0, 16.0);
                push(header, plain(header, solid(0.28, 0.32, 0.42)));
                for index in 0..3u32 {
                    let port = rect(x - 4.0, y + 30.0 + index as f32 * 12.0, 8.0, 8.0);
                    push(
                        port,
                        Quad {
                            origin: [port.min_x, port.min_y],
                            size: [port.width(), port.height()],
                            background: solid(0.72, 0.66, 0.30),
                            border_color: [0.0, 0.0, 0.0, 0.0],
                            corner_radius: 4.0,
                            border_width: 0.0,
                        },
                    );
                }

                // Wires. This is the content §4.3 names as needing a rule, and
                // a real graph has both lengths: most connections are to the
                // next column along and fit inside a tile, and some skip
                // several columns and cannot fit in any tile at all. Both are
                // generated, because the placement rule has a different answer
                // for each and a workload with only one kind would leave half of
                // it untested.
                if spec.wires && column + 1 < spec.columns {
                    let [next_x, next_y] = origin_of(column + 1, row);
                    let start_x = x + GRAPH_NODE_WIDTH;
                    let wire = rect(
                        start_x,
                        y + 34.0,
                        (next_x - start_x).max(1.0),
                        (next_y - y).abs().max(2.0),
                    );
                    push(wire, plain(wire, translucent(0.60, 0.66, 0.78, 0.8)));
                }
                if spec.wires && column % 7 == 0 && column + 4 < spec.columns {
                    let [far_x, far_y] = origin_of(column + 4, row);
                    let start_x = x + GRAPH_NODE_WIDTH;
                    let wire = rect(
                        start_x,
                        y + 46.0,
                        (far_x - start_x).max(1.0),
                        (far_y - y).abs().max(2.0),
                    );
                    push(wire, plain(wire, translucent(0.70, 0.60, 0.50, 0.8)));
                }
            }
        }
        NodeGraph { quads }
    }

    /// Every primitive whose bounds intersect `region`, in plane order.
    fn quads_in(&self, region: Rect) -> Vec<(Rect, Quad)> {
        self.quads
            .iter()
            .filter(|(bounds, _)| bounds.intersects(&region))
            .copied()
            .collect()
    }
}

impl TiledCanvasDriver {
    /// A canvas over an empty scene, at the identity pan.
    pub fn new(spec: &NodeGraphSpec, viewport: Rect, policy: BoundaryPolicy) -> TiledCanvasDriver {
        TiledCanvasDriver {
            scene: Scene::new(),
            compositor: Compositor::new(),
            boundary: BoundaryId::from_raw(1),
            policy,
            viewport,
            graph: NodeGraph::build(spec),
            resident: HashMap::new(),
            frame: 0,
        }
    }

    /// The policy a node-graph canvas is declared with by default.
    pub fn tiled_policy(tile_edge: f32, retain_radius: u32, budget: usize) -> BoundaryPolicy {
        BoundaryPolicy {
            buffering: Buffering::Tiled {
                tile_size: Size::pixels(tile_edge, tile_edge),
                retain_radius,
            },
            resident_tile_budget: budget,
            ..BoundaryPolicy::default()
        }
    }

    /// Pan the canvas so its content composites at `translation`, and bring the
    /// scene up to date. Returns what that frame cost.
    ///
    /// The order is the one a real frame would run in: move, resolve visibility,
    /// create the revealed tiles' layers, offer every visible tile its content,
    /// slide every visible tile, then release what was evicted.
    pub fn pan_to(&mut self, translation: [f32; 2]) -> Result<TiledFrameStats, PatchError> {
        self.frame += 1;
        let frame = self.frame;
        let transform = LayerTransform::translated(translation[0], translation[1]);
        // Declared before it is moved, and the order matters on the first frame
        // only: `Compositor::set_transform` is documented to report `false`
        // rather than create a boundary, so positioning one the compositor has
        // never seen is inert. Written the other way round, this driver's first
        // frame silently resolved its tile span at the identity and the next
        // frame's ordinary pan then looked like a tile crossing.
        self.compositor.visit(self.boundary, self.policy, frame);
        self.compositor.set_transform(self.boundary, transform);

        let Some(visit) =
            self.compositor
                .visit_tiled(self.boundary, self.policy, frame, self.viewport)
        else {
            return Ok(TiledFrameStats::default());
        };

        let mut stats = TiledFrameStats {
            revealed: visit.revealed.clone(),
            visible_tiles: visit.visible.len(),
            resident_tiles: visit.resident,
            evicted: visit.evicted.len(),
            over_budget: visit.over_budget,
            ..TiledFrameStats::default()
        };

        // Layers for tiles that became visible. A brand-new layer starts fully
        // invalidated on `LayerTable::insert`'s own rule — nothing here marks a
        // tile dirty, which is §4.3's point that this is ordinary machinery.
        for tile in &visit.revealed {
            let key = LayerKey::tiled(self.boundary, *tile);
            if !self.scene.layers.contains(LayerId::from_key(key)) {
                stats.layers_created += 1;
            }
            self.scene.layer(key);
        }
        let overlay = visit.overlay_layer();
        if !self.scene.layers.contains(overlay) {
            stats.layers_created += 1;
        }
        self.scene.layer(LayerKey::untiled(self.boundary));

        // Offer every visible tile its content, plus the overlay. A resident
        // tile's content is unchanged, so this produces no patch at all — which
        // is the property the gate reads, established by the protocol rather
        // than by this loop declining to run.
        let mut patch = ScenePatch::new();
        let mut written = 0usize;
        for tile in &visit.visible {
            let layer = visit.tile_layer(*tile);
            let bounds = visit.grid.tile_bounds(*tile);
            let quads: Vec<Quad> = self
                .graph
                .quads_in(bounds)
                .into_iter()
                .filter(|(quad_bounds, _)| visit.placement_layer(*quad_bounds) == layer)
                .map(|(_, quad)| quad)
                .collect();
            written += self.stage_layer(&mut patch, layer, &quads);
        }
        let overlay_region = visit.grid.bounds_of(&visit.visible);
        let overlay_quads: Vec<Quad> = self
            .graph
            .quads_in(overlay_region)
            .into_iter()
            .filter(|(quad_bounds, _)| visit.placement_layer(*quad_bounds) == overlay)
            .map(|(_, quad)| quad)
            .collect();
        let overlay_written = self.stage_layer(&mut patch, overlay, &overlay_quads);
        written += overlay_written;

        let plan = apply(&mut self.scene, &patch)?;
        stats.primitives_written = written;
        stats.overlay_primitives_written = overlay_written;
        stats.upload_bytes = plan.byte_count();

        // Slide every visible tile, and the overlay with them. `set_transform`
        // raises TRANSFORM and nothing else, and is inert when the layer is
        // already there — so a frame that did not move counts zero.
        for layer in visit.visible_layers().into_iter().chain(Some(overlay)) {
            let before = self.scene.layers.get(layer).map(Layer::transform);
            self.scene.layers.set_transform(layer, transform);
            if before != Some(transform) {
                stats.transform_updates += 1;
            }
        }

        for evicted in &visit.evicted {
            let layer = visit.tile_layer(evicted.coord);
            if self.scene.remove_layer(layer) {
                stats.layers_removed += 1;
                self.resident.remove(&layer);
            }
        }

        for layer in self.scene.layers.ids() {
            let Some(record) = self.scene.layers.get(layer) else {
                continue;
            };
            let invalidation = record.invalidation();
            if invalidation.contains(Invalidation::DISPLAY) {
                stats.display_layers.push(layer);
            } else if invalidation == Invalidation::TRANSFORM {
                stats.transform_only_layers += 1;
            }
        }
        stats.display_layers.sort_unstable();
        Ok(stats)
    }

    /// Mark every layer clean, as the end of a rendered frame would.
    pub fn settle(&mut self) {
        for layer in self.scene.layers.ids() {
            self.scene.layers.mark_clean(layer);
        }
    }

    /// Stage one layer's content, returning how many primitives were actually
    /// written — inserted or changed, never merely offered.
    fn stage_layer(&mut self, patch: &mut ScenePatch, layer: LayerId, quads: &[Quad]) -> usize {
        let resident = self.resident.get(&layer).copied().unwrap_or(0);
        let mut written = 0usize;
        let shared = resident.min(quads.len());
        for position in 0..shared {
            let key = RecordKey::from_raw(position as u64 + 1);
            let Some(quad) = quads.get(position) else {
                continue;
            };
            if self.scene.quads.get(layer, key) != Some(quad) {
                patch.quads.update(layer, key, *quad);
                written += 1;
            }
        }
        for position in shared..quads.len() {
            let Some(quad) = quads.get(position) else {
                continue;
            };
            patch.quads.append(
                layer,
                RecordKey::from_raw(position as u64 + 1),
                u32::try_from(position).unwrap_or(u32::MAX),
                *quad,
            );
            written += 1;
        }
        for position in (quads.len()..resident).rev() {
            patch
                .quads
                .remove(layer, RecordKey::from_raw(position as u64 + 1));
            written += 1;
        }
        self.resident.insert(layer, quads.len());
        written
    }

    /// How many primitives the whole scene currently holds.
    pub fn resident_primitives(&self) -> usize {
        self.resident.values().sum()
    }

    /// How many primitives sit on the unbuffered overlay layer — the honest
    /// price of [`crate::scene::TilePlacement::Overlay`].
    pub fn overlay_primitives(&self) -> usize {
        self.resident
            .get(&LayerId::from_key(LayerKey::untiled(self.boundary)))
            .copied()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::occlusion::keep_mask;

    #[test]
    fn the_estimate_bounds_a_frame_without_wildly_overshooting_it() {
        let spec = UiSceneSpec::small();
        let frame = build_frame("initial", &spec);
        let estimate = quad_count_estimate(&spec);
        let actual = frame.quads.len() as u32;
        assert!(
            actual <= estimate,
            "estimate {estimate} is below the actual {actual}, so it is not a bound"
        );
        assert!(
            actual * 2 >= estimate,
            "estimate {estimate} is more than twice the actual {actual}, so it is useless"
        );
    }

    #[test]
    fn the_scene_the_driver_builds_reads_back_as_the_frame_that_built_it() {
        let mut driver = SceneDriver::new();
        let mut frames_that_uploaded = 0;
        let walk = scripted_walk(&UiSceneSpec::small());
        for frame in &walk {
            let plan = driver
                .apply_frame(&frame.quads)
                .expect("applying a generated frame must succeed");
            assert_eq!(
                driver.resident_quads(),
                frame.quads,
                "frame {} did not round-trip through the scene",
                frame.label
            );
            if !plan.is_empty() {
                frames_that_uploaded += 1;
            }
        }
        // Not every frame uploads: closing the modal removes primitives from
        // the very end of the layer, and the protocol's own contract is that
        // shrinking leaves no reachable stale bytes to rewrite. Most frames do,
        // and a harness where none did would be measuring nothing.
        assert!(
            frames_that_uploaded * 2 > walk.len(),
            "only {frames_that_uploaded} of {} frames changed the scene",
            walk.len()
        );
    }

    #[test]
    fn an_unchanged_frame_costs_the_scene_nothing() {
        let frame = build_frame("initial", &UiSceneSpec::small());
        let mut driver = SceneDriver::new();
        driver
            .apply_frame(&frame.quads)
            .expect("first frame applies");
        let repeat = driver
            .apply_frame(&frame.quads)
            .expect("second frame applies");
        assert!(repeat.is_empty(), "an identical frame must upload zero bytes");
        assert_eq!(repeat.byte_count(), 0);
    }

    #[test]
    fn the_walk_actually_produces_something_to_cull() {
        // A harness that culls nothing would pass the differential gate
        // vacuously. Every frame must have real work for the pass to do.
        for frame in scripted_walk(&UiSceneSpec::small()) {
            let keep = keep_mask(&frame.coverage_items(), &frame.poison);
            let culled = keep.iter().filter(|kept| !**kept).count();
            assert!(
                culled > 0,
                "frame {} culls nothing, so it tests nothing",
                frame.label
            );
        }
    }

    #[test]
    fn the_filter_chip_does_not_poison_the_scenes_real_occluders() {
        // The chip's placement is load-bearing (see `build_frame`). If it ever
        // drifts back over the inspector, every cull in the harness quietly
        // disappears and the differential gate starts passing vacuously — so
        // the property is asserted rather than left to the comment.
        let spec = UiSceneSpec::small();
        let frame = build_frame("initial", &spec);
        let with_filter = keep_mask(&frame.coverage_items(), &frame.poison);
        let without_filter = keep_mask(&frame.coverage_items(), &[]);
        let culled_with = with_filter.iter().filter(|kept| !**kept).count();
        let culled_without = without_filter.iter().filter(|kept| !**kept).count();
        assert!(
            culled_with * 2 > culled_without,
            "the filter chip suppressed {} of {culled_without} culls, so it is over an occluder",
            culled_without - culled_with
        );
    }

    #[test]
    fn the_walk_exercises_every_rejection_reason_r_n_8_3_names() {
        let frame = build_frame("initial", &UiSceneSpec::small());
        let items = frame.coverage_items();
        assert!(
            items.iter().any(|item| item.opaque.is_none()),
            "no translucent primitive: the alpha rule is untested"
        );
        assert!(
            frame.quads.iter().any(|quad| quad.corner_radius > 0.0
                && quad.background[3] >= 1.0),
            "no rounded opaque primitive: the corner-radius inset is untested"
        );
        assert!(
            frame.quads.iter().any(|quad| quad.border_width > 0.0
                && quad.border_color[3] < 1.0
                && quad.background[3] >= 1.0),
            "no translucent-bordered opaque primitive: the border inset is untested"
        );
        assert!(
            !frame.poison.is_empty(),
            "no filter region: poisoning and the blur margin are untested"
        );
    }

    fn canvas() -> TiledCanvasDriver {
        TiledCanvasDriver::new(
            &NodeGraphSpec::large(),
            rect(0.0, 0.0, 1024.0, 768.0),
            TiledCanvasDriver::tiled_policy(TileGrid::DEFAULT_EDGE, 1, 256),
        )
    }

    /// §8's Phase 4.5 gate, second clause: *panning within the resident grid
    /// costs one `TRANSFORM` update per visible tile and zero render/reconcile/
    /// layout work anywhere.*
    ///
    /// The pan is deliberately sub-tile — 24px on a 256px grid — and repeated,
    /// so the frame genuinely moves without the visible span changing. The
    /// harness offers every visible tile its content on every one of these
    /// frames (see [`TiledCanvasDriver`]); the zeros below are what the patch
    /// protocol makes of that offer, not what the harness declined to do.
    ///
    /// **The baseline pan is 8px, not zero, and that is not cosmetic.** A
    /// 1024×768 viewport at the identity has both its right and bottom edges on
    /// exact multiples of 256, so it starts perfectly tile-aligned and the very
    /// first pixel of any pan reveals a whole new column — which is a tile
    /// crossing, i.e. the *other* gate. Starting 8px into a tile is what makes
    /// the eight steps below land inside the resident grid, and it is the
    /// condition the first assertion in the loop checks rather than assumes.
    #[test]
    fn gate_panning_inside_the_resident_grid_is_transform_only() -> Result<(), PatchError> {
        let mut canvas = canvas();
        let first = canvas.pan_to([-8.0, -8.0])?;
        assert!(
            first.layers_created > 0 && first.primitives_written > 0,
            "the first frame must actually render, or the comparison is empty"
        );
        canvas.settle();

        for step in 1..=8u32 {
            let stats = canvas.pan_to([-8.0 - step as f32 * 24.0, -8.0])?;
            assert_eq!(
                stats.revealed,
                Vec::<TileCoord>::new(),
                "step {step} crossed a tile boundary, so it is not this gate's case"
            );
            assert_eq!(
                stats.primitives_written, 0,
                "step {step} re-rendered {} primitives for a pan",
                stats.primitives_written
            );
            assert_eq!(stats.upload_bytes, 0, "step {step} uploaded bytes for a pan");
            assert_eq!(stats.layers_created, 0);
            assert_eq!(
                stats.display_layers,
                Vec::<LayerId>::new(),
                "step {step} left a layer needing re-display"
            );
            assert_eq!(
                stats.transform_updates,
                stats.visible_tiles + 1,
                "one TRANSFORM per visible tile, plus the overlay"
            );
            assert_eq!(
                stats.transform_only_layers,
                stats.transform_updates,
                "every layer this frame touched carries TRANSFORM and nothing else"
            );
            assert!(stats.is_transform_only());
            canvas.settle();
        }
        Ok(())
    }

    /// §8's Phase 4.5 gate, first clause: *panning across a tile boundary
    /// renders only the newly-revealed tile(s) — measured directly, not
    /// inferred.*
    ///
    /// "Measured directly" is the load-bearing phrase, so the assertion is not
    /// that the count is small — it is that the set of *tile* layers carrying
    /// `DISPLAY` after the frame is exactly the set belonging to the revealed
    /// tiles, no more and no fewer.
    ///
    /// # One layer beyond the revealed tiles re-displays, and it is the overlay
    ///
    /// This is what running the gate found rather than what writing it assumed.
    /// A crossing dirties the revealed tiles *and the unbuffered overlay*,
    /// because a wire reaching into the newly-revealed column is content that
    /// spans tiles, and [`crate::scene::TilePlacement`]'s rule puts spanning
    /// content on the overlay. So the gate's "only the newly-revealed tile(s)"
    /// is exactly true of the tile grid and not quite true of the boundary: the
    /// overlay is a third thing, and it re-renders whenever the visible region's
    /// spanning content changes.
    ///
    /// The test asserts both halves separately rather than widening the first
    /// one to fit — the tile set is exact, and the overlay's share of the
    /// frame's work is measured and bounded, because an overlay that re-rendered
    /// most of the scene on every crossing would defeat the mechanism while
    /// still passing an "exactly the revealed tiles" check on the tiles alone.
    #[test]
    fn gate_crossing_a_tile_boundary_renders_only_the_revealed_tiles() -> Result<(), PatchError> {
        let mut canvas = canvas();
        canvas.pan_to([-8.0, -8.0])?;
        canvas.settle();
        let resident_after_first = canvas.resident_primitives();

        // A whole tile's worth of pan to the left, so a new column is revealed
        // on the right and nothing else changes.
        let stats = canvas.pan_to([-8.0 - TileGrid::DEFAULT_EDGE, -8.0])?;
        assert!(
            !stats.revealed.is_empty(),
            "the pan did not cross a tile boundary, so this gate tested nothing"
        );
        assert!(
            stats.revealed.len() < stats.visible_tiles,
            "a whole-tile pan revealed {} of {} visible tiles, which is a refill, \
             not a tile crossing",
            stats.revealed.len(),
            stats.visible_tiles
        );

        let overlay = LayerId::from_key(LayerKey::untiled(canvas.boundary));
        let revealed_layers: std::collections::BTreeSet<LayerId> = stats
            .revealed
            .iter()
            .map(|tile| LayerId::from_key(LayerKey::tiled(canvas.boundary, *tile)))
            .collect();
        let displayed_tiles: std::collections::BTreeSet<LayerId> = stats
            .display_layers
            .iter()
            .copied()
            .filter(|layer| *layer != overlay)
            .collect();
        assert_eq!(
            displayed_tiles, revealed_layers,
            "exactly the revealed tiles re-displayed — no more, and no fewer"
        );
        assert!(
            stats.primitives_written > 0,
            "the revealed tiles rendered nothing, so the crossing is not real"
        );

        // The overlay half, disclosed rather than folded into the count above.
        let tile_written = stats.primitives_written - stats.overlay_primitives_written;
        assert!(
            tile_written > 0,
            "every primitive the crossing wrote landed on the overlay, which \
             would mean the tile grid is not carrying the content at all"
        );
        assert!(
            stats.overlay_primitives_written < tile_written,
            "the overlay wrote {} primitives against the tiles' {tile_written}; \
             spanning content has become the dominant cost of a crossing, which \
             is the failure mode TilePlacement's doc discloses",
            stats.overlay_primitives_written
        );
        assert!(
            canvas.resident_primitives() > resident_after_first,
            "the newly-revealed tiles added residency"
        );
        Ok(())
    }

    /// The comparison that makes the crossing a *win* rather than a number: the
    /// same pan under `Buffering::Margin`'s rule re-renders the whole buffered
    /// region, which is §4.3's stated complaint about it.
    ///
    /// Modelled rather than run through a second mechanism, and the model is the
    /// honest one — R-N §7's refill is "the *entire* viewport+margin region
    /// re-renders", so the comparison is against the primitive count of the
    /// whole visible span. That count is read out of this same scene, so the
    /// ratio is a fact about this workload rather than a quoted constant.
    #[test]
    fn a_tile_crossing_renders_a_small_fraction_of_what_a_margin_refill_would()
    -> Result<(), PatchError> {
        let mut canvas = canvas();
        canvas.pan_to([-8.0, -8.0])?;
        canvas.settle();
        let whole_region = canvas.resident_primitives();

        let stats = canvas.pan_to([-8.0 - TileGrid::DEFAULT_EDGE, -8.0])?;
        assert!(
            stats.primitives_written * 4 < whole_region,
            "a crossing wrote {} primitives against a refill's {whole_region}; \
             tiling is supposed to be much better than that, not marginally",
            stats.primitives_written
        );
        Ok(())
    }

    /// The overlay's price, measured rather than asserted — see
    /// [`crate::scene::TilePlacement`] for why spanning content goes there.
    #[test]
    fn wires_are_what_lands_on_the_overlay_and_nodes_are_not() -> Result<(), PatchError> {
        let viewport = rect(0.0, 0.0, 1024.0, 768.0);
        let policy = TiledCanvasDriver::tiled_policy(TileGrid::DEFAULT_EDGE, 1, 256);

        let mut with_wires =
            TiledCanvasDriver::new(&NodeGraphSpec::large(), viewport, policy);
        with_wires.pan_to([0.0, 0.0])?;

        let mut without_wires = TiledCanvasDriver::new(
            &NodeGraphSpec {
                wires: false,
                ..NodeGraphSpec::large()
            },
            viewport,
            policy,
        );
        without_wires.pan_to([0.0, 0.0])?;

        assert!(
            with_wires.overlay_primitives() > without_wires.overlay_primitives(),
            "wires are the spanning content; if they do not reach the overlay, \
             the placement rule is not being exercised"
        );
        assert!(
            with_wires.overlay_primitives() * 3 < with_wires.resident_primitives(),
            "the overlay holds {} of {} primitives — the unbuffered layer has \
             become most of the scene, which is the failure mode this rule's \
             doc discloses",
            with_wires.overlay_primitives(),
            with_wires.resident_primitives()
        );
        Ok(())
    }

    #[test]
    fn a_pan_far_enough_to_leave_the_grid_evicts_the_tiles_behind_it()
    -> Result<(), PatchError> {
        let mut canvas = TiledCanvasDriver::new(
            &NodeGraphSpec::large(),
            rect(0.0, 0.0, 1024.0, 768.0),
            BoundaryPolicy {
                evict_after_frames: 2,
                ..TiledCanvasDriver::tiled_policy(TileGrid::DEFAULT_EDGE, 1, 256)
            },
        );
        canvas.pan_to([0.0, 0.0])?;
        canvas.settle();
        let mut evicted_total = 0usize;
        let mut removed_total = 0usize;
        for step in 1..=12u32 {
            let stats = canvas.pan_to([-(step as f32) * 512.0, 0.0])?;
            evicted_total += stats.evicted;
            removed_total += stats.layers_removed;
            canvas.settle();
        }
        assert!(
            evicted_total > 0,
            "panning right across six tiles left everything behind it resident"
        );
        assert_eq!(
            evicted_total, removed_total,
            "every evicted tile must actually release its layer"
        );
        Ok(())
    }

    #[test]
    fn the_rounded_icons_opaque_region_is_inset_by_its_radius() {
        let frame = build_frame("initial", &UiSceneSpec::small());
        let inset = frame
            .quads
            .iter()
            .zip(frame.coverage_items())
            .find(|(quad, _)| quad.corner_radius == 4.0 && quad.size == [14.0, 14.0])
            .and_then(|(_, item)| item.opaque);
        assert_eq!(
            inset.map(|region| (region.width(), region.height())),
            Some((6.0, 6.0)),
            "a 14px icon with a 4px radius has a 6px opaque core"
        );
    }
}
