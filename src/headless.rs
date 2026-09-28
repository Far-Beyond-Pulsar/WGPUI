//! Test support: render a test window through the real GPU renderer and read
//! the pixels back.
//!
//! [`HeadlessWindow`] attaches a headless `WgpuRenderer` to a test window.
//! Every scene the window draws is rendered by that one long-lived renderer,
//! so slab residency, transform slots, layer textures and the sprite atlas
//! carry across frames exactly as they do on screen, and the renderer's
//! re-record requests flow back the way a platform window's do. Comparing a
//! window driven normally against one fully re-rendered every frame turns
//! "a panel blinks after some interaction" into a failing test.
//!
//! Pair with [`TestAppContext::with_real_text_system`] so text is shaped and
//! drawn as real glyph sprites.

use crate::platform::cross::{
    atlas::WgpuAtlas,
    render_context::{WgpuContext, WgpuOptions},
    renderer::WgpuRenderer,
};
use crate::{AnyWindowHandle, InputEvent, Modifiers, MouseButton, Pixels, Point, TestAppContext};
use std::{cell::RefCell, rc::Rc, sync::Arc};

/// A test window whose frames are rendered by a real GPU renderer.
pub struct HeadlessWindow {
    handle: AnyWindowHandle,
    renderer: Rc<RefCell<WgpuRenderer>>,
    presented: Rc<RefCell<Vec<u8>>>,
}

impl HeadlessWindow {
    /// Attach a renderer to `window` and draw it once in full. `None` when no
    /// GPU adapter is available; callers should skip.
    pub fn attach(window: impl Into<AnyWindowHandle>, cx: &mut TestAppContext) -> Option<Self> {
        let handle = window.into();
        let context = Arc::new(WgpuContext::new(&WgpuOptions::default()).ok()?);
        let (width, height) = handle
            .update(cx, |_, window, _| {
                let device = window.viewport_size().scale(window.scale_factor());
                (device.width.0.round() as u32, device.height.0.round() as u32)
            })
            .ok()?;
        let atlas = Arc::new(WgpuAtlas::new(context.clone()));
        let renderer = Rc::new(RefCell::new(WgpuRenderer::new_headless(
            context,
            atlas.clone(),
            width,
            height,
        )));
        let presented = Rc::new(RefCell::new(Vec::new()));
        handle
            .update(cx, |_, window, _| {
                window.sprite_atlas = atlas;
                let renderer = renderer.clone();
                let presented = presented.clone();
                window.test_frame_sink = Some(Box::new(move |scene| {
                    let mut renderer = renderer.borrow_mut();
                    assert_draw_order(scene);
                    renderer.draw(scene);
                    *presented.borrow_mut() = renderer.read_back_frame();
                    (renderer.take_rerecord_requests(), renderer.take_dead_page_requests())
                }));
                window.refresh();
            })
            .ok()?;
        cx.run_until_parked();
        let this = Self {
            handle,
            renderer,
            presented,
        };
        // Scroll containers measure themselves on their first frame; start
        // from a settled state.
        this.draw(cx, true);
        Some(this)
    }

    pub fn handle(&self) -> AnyWindowHandle {
        self.handle
    }

    /// Device-pixel size of the frames.
    pub fn size(&self) -> (u32, u32) {
        self.renderer.borrow().frame_size()
    }

    /// The last frame presented, as tightly packed RGBA8 rows.
    pub fn presented(&self) -> Vec<u8> {
        self.presented.borrow().clone()
    }

    /// A real frame, as the platform frame loop draws one. `full` bypasses
    /// every retained layer and view cache: the reference render.
    pub fn draw(&self, cx: &mut TestAppContext, full: bool) {
        self.handle
            .update(cx, |_, window, cx| {
                window.route_renderer_requests();
                if full {
                    window.refresh();
                }
                window.refresh_buffers();
                window.draw(cx).clear();
            })
            .expect("window update");
        cx.run_until_parked();
    }

    /// An idle frame where only an external surface (the viewport) changed:
    /// exactly the platform frame callback's decision -- pending renderer
    /// requests routed first, the previous scene re-presented when nothing
    /// else is pending, a real draw otherwise.
    pub fn idle_frame(&self, cx: &mut TestAppContext) -> Vec<u8> {
        let needs_draw = self
            .handle
            .update(cx, |_, window, cx| {
                window.route_renderer_requests();
                window.refresh_buffers();
                if !window.present_previous_scene_if_display_only(cx) {
                    return true;
                }
                let mut renderer = self.renderer.borrow_mut();
                renderer.draw(&window.rendered_frame.scene);
                window.test_pending_rerecords.extend(renderer.take_rerecord_requests());
                window.test_pending_dead_pages.extend(renderer.take_dead_page_requests());
                *self.presented.borrow_mut() = renderer.read_back_frame();
                false
            })
            .expect("window update");
        if needs_draw {
            self.draw(cx, false);
        }
        self.presented()
    }

    /// The renderer's own wake-ups with nothing else happening: while it has
    /// requests pending, the frame it scheduled routes them and draws. Every
    /// request must mark the window dirty and be answered within a few
    /// frames. Returns how many wake frames it took.
    pub fn settle(&self, cx: &mut TestAppContext) -> Result<usize, String> {
        for wakes in 0..4 {
            let (pending, dirty) = self
                .handle
                .update(cx, |_, window, _| {
                    let pending = !window.test_pending_rerecords.is_empty()
                        || !window.test_pending_dead_pages.is_empty();
                    window.route_renderer_requests();
                    (pending, window.invalidator.is_dirty())
                })
                .expect("window update");
            if !pending {
                return Ok(wakes);
            }
            if !dirty {
                return Err("renderer requests were pending but routing them left the window \
                            clean: skipped content would stay blank until an unrelated redraw"
                    .to_string());
            }
            self.handle
                .update(cx, |_, window, cx| window.draw(cx).clear())
                .expect("window update");
            cx.run_until_parked();
        }
        Err("renderer requests were still pending after 4 wake frames".to_string())
    }

    /// Where the element tagged `.debug_selector(|| selector)` was painted,
    /// in logical pixels. Recorded only when the element actually paints, so
    /// query after a full [`Self::draw`].
    pub fn element_bounds(
        &self,
        cx: &mut TestAppContext,
        selector: &str,
    ) -> Option<crate::Bounds<Pixels>> {
        self.handle
            .update(cx, |_, window, _| window.rendered_frame.debug_bounds.get(selector).copied())
            .ok()
            .flatten()
    }

    /// Move the pointer to `position` (logical pixels).
    pub fn mouse_move(&self, cx: &mut TestAppContext, position: Point<Pixels>) {
        cx.test_window(self.handle).simulate_input(
            crate::MouseMoveEvent {
                position,
                modifiers: Modifiers::default(),
                pressed_button: None,
            }
            .to_platform_input(),
        );
        cx.run_until_parked();
    }

    /// Press and release the left button at `position` (logical pixels).
    pub fn click(&self, cx: &mut TestAppContext, position: Point<Pixels>) {
        cx.test_window(self.handle).simulate_input(
            crate::MouseDownEvent {
                button: MouseButton::Left,
                position,
                modifiers: Modifiers::default(),
                click_count: 1,
                first_mouse: false,
            }
            .to_platform_input(),
        );
        cx.run_until_parked();
        cx.test_window(self.handle).simulate_input(
            crate::MouseUpEvent {
                button: MouseButton::Left,
                position,
                modifiers: Modifiers::default(),
                click_count: 1,
            }
            .to_platform_input(),
        );
        cx.run_until_parked();
    }
}

/// Compare two frames. On a mismatch, `live.png`, `truth.png` and `diff.png`
/// (differing pixels in magenta) are written under
/// `<crate>/target/layer-flicker/<label>/` and the error names the rect.
pub fn compare_frames(
    live: &[u8],
    truth: &[u8],
    width: u32,
    height: u32,
    label: &str,
    output_root: &std::path::Path,
) -> Result<(), String> {
    if live.len() != truth.len() {
        return Err(format!("{label}: frame sizes differ ({} vs {})", live.len(), truth.len()));
    }
    let mut differing = 0usize;
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (u32::MAX, u32::MAX, 0, 0);
    let mut diff = vec![0u8; live.len()];
    for (index, (a, b)) in live.chunks(4).zip(truth.chunks(4)).enumerate() {
        let delta = a.iter().zip(b).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
        if delta > 3 {
            differing += 1;
            let (x, y) = (index as u32 % width, index as u32 / width);
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            diff[index * 4..index * 4 + 4].copy_from_slice(&[255, 0, 255, 255]);
        } else {
            diff[index * 4..index * 4 + 4].copy_from_slice(&[b[0] / 3, b[1] / 3, b[2] / 3, 255]);
        }
    }
    if differing == 0 {
        return Ok(());
    }
    let folder: String = label
        .chars()
        .map(|character| if character.is_ascii_alphanumeric() { character } else { '_' })
        .collect();
    let dir = output_root.join("layer-flicker").join(folder);
    let saved = std::fs::create_dir_all(&dir).is_ok()
        && [("live.png", live), ("truth.png", truth), ("diff.png", &diff[..])]
            .iter()
            .all(|(name, pixels)| {
                image::RgbaImage::from_raw(width, height, pixels.to_vec())
                    .is_some_and(|image| image.save(dir.join(name)).is_ok())
            });
    Err(format!(
        "{label}: {differing} pixels differ from a full render, in device rect \
         ({min_x},{min_y})..=({max_x},{max_y}){}",
        if saved {
            format!("; images in {}", dir.display())
        } else {
            String::new()
        }
    ))
}

/// Walk the scene's draw stream exactly as the renderer does and require
/// draw orders never to go backwards between one batch or slab span and the
/// next: whatever resolves to a higher order must also draw later.
pub(crate) fn assert_draw_order(scene: &crate::Scene) {
    use crate::scene::{PrimitiveBatch, SceneBatch};
    fn range<T>(items: &[T], order: impl Fn(&T) -> u32) -> Option<(u32, u32)> {
        let first = order(items.first()?);
        let last = items.iter().map(&order).max()?;
        Some((first, last))
    }
    let mut previous: Option<(u32, String)> = None;
    for (position, batch) in scene.frame_batches().enumerate() {
        let (first, last, what) = match &batch {
            SceneBatch::LayerSlab(index) => {
                let span = &scene.layer_slab_spans[*index];
                (span.order(), span.order(), format!("slab span {index} of {:?}", span.key))
            }
            SceneBatch::Primitives(batch) => {
                let (name, range) = match batch {
                    PrimitiveBatch::Quads(items) => ("quads", range(items, |item| item.order)),
                    PrimitiveBatch::Shadows(items) => ("shadows", range(items, |item| item.order)),
                    PrimitiveBatch::Underlines(items) => {
                        ("underlines", range(items, |item| item.order))
                    }
                    PrimitiveBatch::MonochromeSprites { sprites, .. } => {
                        ("mono sprites", range(sprites, |item| item.order))
                    }
                    PrimitiveBatch::PolychromeSprites { sprites, .. } => {
                        ("poly sprites", range(sprites, |item| item.order))
                    }
                    PrimitiveBatch::Surfaces(items) => ("surfaces", range(items, |item| item.order)),
                    _ => continue,
                };
                let Some((first, last)) = range else {
                    continue;
                };
                (first, last, format!("{name} batch at orders {first}..={last}"))
            }
        };
        if let Some((previous_last, previous_what)) = &previous {
            assert!(
                first >= *previous_last,
                "draw stream out of order at item {position}: {what} (order {first}) draws after \
                 {previous_what}, which reached order {previous_last}"
            );
        }
        previous = Some((last, what));
    }
}
