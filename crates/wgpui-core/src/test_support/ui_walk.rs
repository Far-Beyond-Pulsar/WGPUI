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

use crate::geometry::Rect;
use crate::occlusion::{CoverageItem, PoisonRegion, quad_coverage_item};
use crate::patch::apply::{ScenePatch, UploadPlan, apply};
use crate::patch::primitive::Quad;
use crate::patch::{PatchError, RecordKey};
use crate::scene::{BoundaryId, LayerId, LayerKey, Scene};

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
