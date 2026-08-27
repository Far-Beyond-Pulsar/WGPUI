//! Boundary texture-retention pool (today's `LayerTextureEntry`).
//! See docs/gpu-native-architecture.md §3.5, §5.5.
//!
//! # This is where `.boundary()`'s texture retention stops being a decision
//!
//! `docs/phase-2-results.md` §7 is explicit about what Phase 2 owed and what it
//! did not: "`Retention::Texture` is a decision. No texture pool, no
//! rasterize-to-texture, no composite entry. Phase 4." Phase 2 could not have
//! built it — §3.1 forbids `wgpui-core` a live `wgpu::Device`, and there was no
//! other crate with one until Phase 3. This file is the other half.
//!
//! What a boundary that reached `Retention::Texture` gets here: a persistent
//! offscreen texture of its own, keyed by its `BoundaryId`, kept across frames,
//! re-baked only when its content token changes or its size does, and returned
//! to nothing at all after an idle interval. Its composite entry then reaches
//! the framebuffer through the same path an externally-rendered surface's does
//! (`external_surface.rs`), which is §5.5's Gap 2.
//!
//! # Ported from `LayerTextureEntry`, with its two rules kept
//!
//! `renderer.rs:2190-2201` holds the legacy entry, and two of its properties
//! are load-bearing rather than incidental:
//!
//! 1. **The content token is compared, not the pixels.** An entry records the
//!    generation its texture was baked at; a boundary whose token still matches
//!    composites its existing texture and re-renders nothing. `wgpui-core`
//!    already produces exactly such a number —
//!    `wgpui_core::scene::Layer::generation`, bumped on every reservation,
//!    resize, release, or content edit — so this file introduces no new
//!    invalidation concept, which is what §5.4 asks of anything touching this
//!    area.
//! 2. **Eviction posts a re-record request rather than silently dropping.** The
//!    legacy comment is blunt about why: a layer whose texture vanished must
//!    re-bake on its next composite "instead of sampling a missing texture."
//!    Here [`LayerTexturePool::sweep`] returns the evicted boundaries and
//!    [`LayerTexturePool::acquire`] reports [`Bake::Required`] for a boundary it
//!    has no entry for, so the same obligation is a return value rather than a
//!    side channel.

use std::collections::HashMap;

use wgpui_core::scene::layer::BoundaryId;

use crate::render::pipelines::TARGET_FORMAT;

/// Frames a boundary's texture may go untouched before it is released.
///
/// `LAYER_TEXTURE_IDLE_FRAMES` in the legacy renderer, same value. Deliberately
/// much longer than `BoundaryPolicy::DEFAULT_EVICT_AFTER_FRAMES` (60): losing
/// the *texture* costs a re-bake, losing the boundary's *state* costs its
/// scroll position, and the cheaper loss can afford to happen sooner.
pub const IDLE_FRAMES: u64 = 240;

/// Whether a boundary's texture holds current content.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Bake {
    /// The texture is current — composite it and render nothing.
    Reusable,
    /// The texture is new, resized, or stale, and must be rendered into before
    /// it is composited.
    Required,
}

impl Bake {
    /// Whether the caller has to render into the texture this frame.
    pub const fn is_required(self) -> bool {
        matches!(self, Bake::Required)
    }
}

/// One texture-retained boundary's persistent offscreen texture.
struct Entry {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    content_token: u64,
    last_used_frame: u64,
}

/// What the pool has done, in the style `render_stats` established.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct LayerTextureStats {
    /// Textures created. A steady-state frame loop must not keep raising this.
    pub allocations: u64,
    /// Textures released by [`LayerTexturePool::sweep`].
    pub evictions: u64,
    /// Acquisitions that found a current texture and re-rendered nothing.
    pub reuses: u64,
    /// Acquisitions that had to re-bake.
    pub bakes: u64,
}

/// Every texture-retained boundary's texture, across frames.
pub struct LayerTexturePool {
    entries: HashMap<BoundaryId, Entry>,
    frame: u64,
    idle_frames: u64,
    stats: LayerTextureStats,
}

impl Default for LayerTexturePool {
    fn default() -> Self {
        LayerTexturePool::new(IDLE_FRAMES)
    }
}

impl LayerTexturePool {
    /// A pool that releases a texture after `idle_frames` frames untouched.
    pub fn new(idle_frames: u64) -> LayerTexturePool {
        LayerTexturePool {
            entries: HashMap::new(),
            frame: 0,
            idle_frames,
            stats: LayerTextureStats::default(),
        }
    }

    /// Advance the frame counter. Call once per frame, before any acquisition.
    pub fn begin_frame(&mut self) -> u64 {
        self.frame += 1;
        self.frame
    }

    /// The current frame counter.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// What the pool has done so far.
    pub fn stats(&self) -> LayerTextureStats {
        self.stats
    }

    /// How many textures are live.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the pool holds nothing.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get `boundary`'s texture, creating or recreating it as needed, and say
    /// whether the caller has to render into it.
    ///
    /// A size change recreates rather than resizing in place, exactly as the
    /// legacy entry does: a `wgpu::Texture`'s extent is fixed at creation, and
    /// the alternative — sampling the old texture across new bounds — is the
    /// stretched-frame artefact `SurfaceRegistry::resize` goes to some trouble
    /// to avoid on its own side.
    pub fn acquire(
        &mut self,
        device: &wgpu::Device,
        boundary: BoundaryId,
        width: u32,
        height: u32,
        content_token: u64,
    ) -> Bake {
        let width = width.max(1);
        let height = height.max(1);
        let frame = self.frame;
        if let Some(entry) = self.entries.get_mut(&boundary)
            && entry.width == width
            && entry.height == height
        {
            entry.last_used_frame = frame;
            if entry.content_token == content_token {
                self.stats.reuses += 1;
                return Bake::Reusable;
            }
            entry.content_token = content_token;
            self.stats.bakes += 1;
            return Bake::Required;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("boundary layer texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.entries.insert(
            boundary,
            Entry {
                texture,
                view,
                width,
                height,
                content_token,
                last_used_frame: frame,
            },
        );
        self.stats.allocations += 1;
        self.stats.bakes += 1;
        Bake::Required
    }

    /// A boundary's texture view, for compositing or for rendering into.
    pub fn view(&self, boundary: BoundaryId) -> Option<&wgpu::TextureView> {
        self.entries.get(&boundary).map(|entry| &entry.view)
    }

    /// A boundary's texture, for a readback.
    pub fn texture(&self, boundary: BoundaryId) -> Option<&wgpu::Texture> {
        self.entries.get(&boundary).map(|entry| &entry.texture)
    }

    /// A boundary's texture size.
    pub fn size(&self, boundary: BoundaryId) -> Option<(u32, u32)> {
        self.entries
            .get(&boundary)
            .map(|entry| (entry.width, entry.height))
    }

    /// Release every texture untouched for longer than the idle interval,
    /// returning the boundaries that lost one.
    ///
    /// The return value is the legacy path's "post a re-record request" made
    /// explicit — see this module's doc.
    pub fn sweep(&mut self) -> Vec<BoundaryId> {
        let frame = self.frame;
        let idle = self.idle_frames;
        let mut evicted = Vec::new();
        self.entries.retain(|boundary, entry| {
            if frame.saturating_sub(entry.last_used_frame) <= idle {
                return true;
            }
            evicted.push(*boundary);
            false
        });
        self.stats.evictions += evicted.len() as u64;
        evicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::device::context_or_report;

    const PANEL: BoundaryId = BoundaryId::from_raw(7);

    #[test]
    fn a_boundary_keeps_one_texture_across_frames_and_rebakes_only_when_its_token_moves() {
        let Some(context) = context_or_report("layer_texture_pool") else {
            return;
        };
        let mut pool = LayerTexturePool::new(IDLE_FRAMES);

        pool.begin_frame();
        assert_eq!(
            pool.acquire(&context.device, PANEL, 640, 480, 1),
            Bake::Required,
            "a boundary the pool has never seen must render into its new texture"
        );
        assert!(pool.view(PANEL).is_some());

        pool.begin_frame();
        assert_eq!(
            pool.acquire(&context.device, PANEL, 640, 480, 1),
            Bake::Reusable,
            "an unchanged boundary composites its existing texture and renders \
             nothing — the whole point of Retention::Texture"
        );

        pool.begin_frame();
        assert_eq!(
            pool.acquire(&context.device, PANEL, 640, 480, 2),
            Bake::Required,
            "a changed content token must re-bake"
        );

        pool.begin_frame();
        assert_eq!(
            pool.acquire(&context.device, PANEL, 800, 480, 2),
            Bake::Required,
            "a resize must re-bake, because a texture's extent is fixed"
        );
        assert_eq!(pool.size(PANEL), Some((800, 480)));

        let stats = pool.stats();
        assert_eq!(
            stats.allocations, 2,
            "one texture for the original size, one for the resize — the token \
             change must not have allocated"
        );
        assert_eq!(stats.reuses, 1);
        assert_eq!(stats.bakes, 3);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn an_idle_boundary_loses_its_texture_and_is_named_so_it_can_re_record() {
        let Some(context) = context_or_report("layer_texture_eviction") else {
            return;
        };
        let mut pool = LayerTexturePool::new(2);
        pool.begin_frame();
        pool.acquire(&context.device, PANEL, 64, 64, 1);

        for _ in 0..2 {
            pool.begin_frame();
            assert!(pool.sweep().is_empty(), "still inside the idle interval");
        }
        pool.begin_frame();
        assert_eq!(pool.sweep(), vec![PANEL]);
        assert!(pool.is_empty());
        assert!(
            pool.view(PANEL).is_none(),
            "an evicted boundary must not be able to sample a missing texture"
        );

        pool.begin_frame();
        assert_eq!(
            pool.acquire(&context.device, PANEL, 64, 64, 1),
            Bake::Required,
            "the same token as before must still re-bake, because the texture \
             holding it is gone"
        );
        assert_eq!(pool.stats().evictions, 1);
        assert_eq!(pool.stats().allocations, 2);
    }

    #[test]
    fn a_steady_frame_loop_allocates_nothing_after_the_first_frame() {
        let Some(context) = context_or_report("layer_texture_steady_state") else {
            return;
        };
        let mut pool = LayerTexturePool::new(IDLE_FRAMES);
        for _ in 0..30 {
            pool.begin_frame();
            pool.acquire(&context.device, PANEL, 320, 200, 1);
            assert!(pool.sweep().is_empty());
        }
        assert_eq!(pool.stats().allocations, 1);
        assert_eq!(pool.stats().reuses, 29);
        assert_eq!(pool.stats().bakes, 1);
    }
}
