use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use collections::{FxHashMap, FxHashSet};
use wgpu::util::DeviceExt;
use wgpu::CurrentSurfaceTexture;

use crate::{
    AtlasTextureId, AtlasTile, BackdropFilter, DevicePixels, FilterBoundary, GpuSpecs,
    GradientStop, LayerKey, LinearColorStop, MonochromeSprite, Pixels, PlatformAtlas,
    PrimitiveBatch, Quad, ScaledPixels, Scene, TransformationMatrix, color, geometry,
    platform::cross::{
        atlas::WgpuAtlas,
        render_context::{WgpuContext, ensure_buffer_size},
        slab::{SlabKind, MIN_CLASS},
        slab_gpu::{self, GpuLayerTransform, SlabGpuBuffers, SlabRegistry, SyncPlan},
        surface_registry::SurfaceId,
    },
};

const fn map_attributes<const N: usize>(
    attribs: &'static [wgpu::VertexAttribute; N],
    location_offset: u32,
    offset_offset: wgpu::BufferAddress,
) -> [wgpu::VertexAttribute; N] {
    let mut result = [wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        // NOTE(mdeand): Dummy format, will be overwritten.
        format: wgpu::VertexFormat::Uint8x2,
    }; N];
    let mut i = 0;

    while i < result.len() {
        result[i] = wgpu::VertexAttribute {
            offset: attribs[i].offset + offset_offset,
            shader_location: attribs[i].shader_location + location_offset,
            format: attribs[i].format,
        };
        i += 1;
    }

    result
}

/// Fragment-stage translate-undo edits, per shader: patterns that must occur
/// exactly once in that shader's body and get rewritten to route through
/// `layer_world_position`. Shaders absent from this list read no world-space
/// geometry in their fragment stages and must stay untouched.
const FRAGMENT_TRANSLATE_EDITS: &[(&str, &str, &str)] = &[
    (
        "quads",
        "gradient_color(quad.background, input.position.xy, quad.bounds,",
        "gradient_color(quad.background, layer_world_position(input.position.xy), quad.bounds,",
    ),
    (
        "quads",
        "let point = input.position.xy - quad.bounds.origin;",
        "let point = layer_world_position(input.position.xy) - quad.bounds.origin;",
    ),
    (
        "shadows",
        "let center_to_point = input.position.xy - center;",
        "let center_to_point = layer_world_position(input.position.xy) - center;",
    ),
    (
        "underlines",
        "let st = (input.position.xy - underline.bounds.origin)",
        "let st = (layer_world_position(input.position.xy) - underline.bounds.origin)",
    ),
    (
        "poly_sprites",
        "quad_sdf(input.position.xy, sprite.bounds, sprite.corner_radii)",
        "quad_sdf(layer_world_position(input.position.xy), sprite.bounds, sprite.corner_radii)",
    ),
];

/// Shaders whose vertex stage builds NDC positions through the shared
/// `to_device_position_impl` helper; `paths.wgsl` builds them inline instead.
const IMPL_VERTEX_SHADERS: &[&str] = &[
    "quads",
    "shadows",
    "underlines",
    "mono_sprites",
    "poly_sprites",
];

/// Shader source for a pipeline that can draw spliced layer-slab content:
/// the shared transform-uniform prelude ahead of the file's body, with exact
/// match-once edits threading the per-layer translate through the vertex
/// stage and undoing it in fragment stages that re-read world-space geometry.
///
/// The `.wgsl` files themselves stay byte-pristine: `flamegraph_replay`
/// renders them against its own bind-group layouts, so every slab-specific
/// reference must come from this composition step. Each edit asserts exactly
/// one match — a shader change that drifts past these patterns fails loudly
/// here instead of silently dropping the translate (or double-applying it).
fn slab_shader_source(name: &str, group: u32, body: &'static str) -> std::borrow::Cow<'static, str> {
    let mut source = include_str!("shaders/slab_transform.wgsl")
        .replace("{SLAB_TRANSFORM_GROUP}", &group.to_string());

    // Vertex stage: shift rasterized positions by the layer translate. Clip
    // distances are computed from untranslated bounds on purpose — they move
    // with the instance data, not the rasterized position.
    if IMPL_VERTEX_SHADERS.contains(&name) {
        const PATTERN: &str = "let device_position = position / globals.viewport_size";
        assert!(
            body.matches(PATTERN).count() == 1,
            "{name}: vertex-position pattern drifted"
        );
        source.push_str(&body.replacen(
            PATTERN,
            "let device_position = (position + layer_transform.translate) / globals.viewport_size",
            1,
        ));
    } else {
        const PATTERN: &str = "let device_pos = v.xy_position / globals.viewport_size";
        assert!(
            name == "paths" && body.matches(PATTERN).count() == 1,
            "{name}: no known vertex-position pattern; slab transform edits are stale"
        );
        source.push_str(&body.replacen(
            PATTERN,
            "let world_position = v.xy_position + layer_transform.translate;\n    let device_pos = world_position / globals.viewport_size",
            1,
        ));
    }

    for (shader, pattern, replacement) in FRAGMENT_TRANSLATE_EDITS {
        if *shader != name {
            continue;
        }
        assert_eq!(
            source.matches(pattern).count(),
            1,
            "{name}: fragment edit matched more than once: {pattern}"
        );
        source = source.replace(pattern, replacement);
    }

    std::borrow::Cow::Owned(source)
}

impl color::Hsla {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 4] = &[
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(color::Hsla, h) as wgpu::BufferAddress,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(color::Hsla, s) as wgpu::BufferAddress,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(color::Hsla, l) as wgpu::BufferAddress,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(color::Hsla, a) as wgpu::BufferAddress,
            shader_location: 3,
            format: wgpu::VertexFormat::Float32,
        },
    ];
}

impl color::GradientStop {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 2] = &[
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(GradientStop, color) as wgpu::BufferAddress,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(GradientStop, position) as wgpu::BufferAddress,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32,
        },
    ];
}

impl color::LinearColorStop {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 2] = &[
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(LinearColorStop, color) as wgpu::BufferAddress,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(LinearColorStop, percentage) as wgpu::BufferAddress,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32,
        },
    ];
}

impl color::Background {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 9] = &{
        let linear_color_stop_vertex_attributes = map_attributes(
            GradientStop::VERTEX_ATTRIBUTES,
            7,
            std::mem::offset_of!(color::Background, colors) as wgpu::BufferAddress,
        );

        [
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(color::Background, tag) as wgpu::BufferAddress,
                shader_location: 0,
                format: wgpu::VertexFormat::Uint32,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(color::Background, color_space) as wgpu::BufferAddress,
                shader_location: 1,
                format: wgpu::VertexFormat::Uint32,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(color::Background, solid) as wgpu::BufferAddress,
                shader_location: 2,
                format: wgpu::VertexFormat::Uint32,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(color::Background, param0) as wgpu::BufferAddress,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(color::Background, param1) as wgpu::BufferAddress,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(color::Background, param2) as wgpu::BufferAddress,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(color::Background, param3) as wgpu::BufferAddress,
                shader_location: 6,
                format: wgpu::VertexFormat::Float32,
            },
            linear_color_stop_vertex_attributes[0],
            linear_color_stop_vertex_attributes[1],
            // wgpu::VertexAttribute {
            //     offset: std::mem::offset_of!(color::Background, pad) as wgpu::BufferAddress,
            //     shader_location: 9,
            //     format: wgpu::VertexFormat::Uint8,
            // },
        ]
    };
}

impl color::TextColor {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 7] = &{
        let linear_color_stop_vertex_attributes = map_attributes(
            LinearColorStop::VERTEX_ATTRIBUTES,
            4,
            std::mem::offset_of!(color::TextColor, colors) as wgpu::BufferAddress,
        );

        [
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(color::TextColor, tag) as wgpu::BufferAddress,
                shader_location: 0,
                format: wgpu::VertexFormat::Uint32,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(color::TextColor, color_space) as wgpu::BufferAddress,
                shader_location: 1,
                format: wgpu::VertexFormat::Uint32,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(color::TextColor, solid) as wgpu::BufferAddress,
                shader_location: 2,
                format: wgpu::VertexFormat::Uint32,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(color::TextColor, gradient_angle_or_reserved)
                    as wgpu::BufferAddress,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32,
            },
            linear_color_stop_vertex_attributes[0],
            linear_color_stop_vertex_attributes[1],
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(color::TextColor, pad) as wgpu::BufferAddress,
                shader_location: 6,
                format: wgpu::VertexFormat::Uint32,
            },
        ]
    };
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalParams {
    viewport_size: [f32; 2],
    premultimated_alpha: u32,
    pad: u32,
}

// Size of `Globals` in the uniform address space, where WGSL rounds a struct's
// byte size up to a multiple of its 16-byte binding alignment.
const _: () = assert!(std::mem::size_of::<GlobalParams>() == 16);

impl GlobalParams {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 3] = &[
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(GlobalParams, viewport_size) as wgpu::BufferAddress,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x2,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(GlobalParams, premultimated_alpha) as wgpu::BufferAddress,
            shader_location: 1,
            format: wgpu::VertexFormat::Uint32,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(GlobalParams, pad) as wgpu::BufferAddress,
            shader_location: 2,
            format: wgpu::VertexFormat::Uint32,
        },
    ];
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Bounds {
    origin: [f32; 2],
    size: [f32; 2],
}

impl geometry::Corners<ScaledPixels> {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 4] = &[
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(geometry::Corners<ScaledPixels>, top_left)
                as wgpu::BufferAddress,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(geometry::Corners<ScaledPixels>, top_right)
                as wgpu::BufferAddress,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(geometry::Corners<ScaledPixels>, bottom_right)
                as wgpu::BufferAddress,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(geometry::Corners<ScaledPixels>, bottom_left)
                as wgpu::BufferAddress,
            shader_location: 3,
            format: wgpu::VertexFormat::Float32,
        },
    ];
}

impl geometry::Edges<ScaledPixels> {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 4] = &[
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(geometry::Edges<ScaledPixels>, top) as wgpu::BufferAddress,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(geometry::Edges<ScaledPixels>, right)
                as wgpu::BufferAddress,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(geometry::Edges<ScaledPixels>, bottom)
                as wgpu::BufferAddress,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(geometry::Edges<ScaledPixels>, left)
                as wgpu::BufferAddress,
            shader_location: 3,
            format: wgpu::VertexFormat::Float32,
        },
    ];
}

impl Bounds {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 2] = &[
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(Bounds, origin) as wgpu::BufferAddress,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x2,
        },
        wgpu::VertexAttribute {
            offset: std::mem::offset_of!(Bounds, size) as wgpu::BufferAddress,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x2,
        },
    ];
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SurfaceParams {
    bounds: Bounds,
    content_mask: Bounds,
}

impl Quad {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 22] = &{
        let bounds_vertex_attributes = map_attributes(
            Bounds::VERTEX_ATTRIBUTES,
            2,
            std::mem::offset_of!(Quad, bounds) as wgpu::BufferAddress,
        );

        let content_mask_vertex_attributes = map_attributes(
            Bounds::VERTEX_ATTRIBUTES,
            4,
            std::mem::offset_of!(Quad, content_mask) as wgpu::BufferAddress,
        );

        let background_vertex_attributes = map_attributes(
            color::Background::VERTEX_ATTRIBUTES,
            6,
            std::mem::offset_of!(Quad, background) as wgpu::BufferAddress,
        );

        let border_color_vertex_attributes = map_attributes(
            color::Hsla::VERTEX_ATTRIBUTES,
            11,
            std::mem::offset_of!(Quad, border_color) as wgpu::BufferAddress,
        );

        let corner_radii_vertex_attributes = map_attributes(
            geometry::Corners::<ScaledPixels>::VERTEX_ATTRIBUTES,
            15,
            std::mem::offset_of!(Quad, corner_radii) as wgpu::BufferAddress,
        );

        let border_widths_vertex_attributes = map_attributes(
            geometry::Edges::<ScaledPixels>::VERTEX_ATTRIBUTES,
            19,
            std::mem::offset_of!(Quad, border_widths) as wgpu::BufferAddress,
        );

        [
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(Quad, order) as wgpu::BufferAddress,
                shader_location: 0,
                format: wgpu::VertexFormat::Uint32,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(Quad, border_style) as wgpu::BufferAddress,
                shader_location: 1,
                format: wgpu::VertexFormat::Uint32,
            },
            bounds_vertex_attributes[0],
            bounds_vertex_attributes[1],
            content_mask_vertex_attributes[0],
            content_mask_vertex_attributes[1],
            background_vertex_attributes[0],
            background_vertex_attributes[1],
            background_vertex_attributes[2],
            background_vertex_attributes[3],
            border_color_vertex_attributes[0],
            border_color_vertex_attributes[1],
            border_color_vertex_attributes[2],
            border_color_vertex_attributes[3],
            corner_radii_vertex_attributes[0],
            corner_radii_vertex_attributes[1],
            corner_radii_vertex_attributes[2],
            corner_radii_vertex_attributes[3],
            border_widths_vertex_attributes[0],
            border_widths_vertex_attributes[1],
            border_widths_vertex_attributes[2],
            border_widths_vertex_attributes[3],
        ]
    };
}

#[repr(C)]
struct QuadsData {
    globals: GlobalParams,
}

#[repr(C)]
struct ShadowsData {
    globals: GlobalParams,
}

#[repr(C)]
struct PathRasterizationData {
    globals: GlobalParams,
}

struct PathsData {
    globals: GlobalParams,
    t_sprite: wgpu::TextureView,
    s_sprite: wgpu::Sampler,
}

/// Per-vertex data uploaded to the GPU for path rendering.
/// Layout must exactly match the `GpuPathVertex` struct in `paths.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPathVertex {
    xy_position: [f32; 2],         // offset  0
    st_position: [f32; 2],         // offset  8
    hsla: [f32; 4],                // offset 16  (h, s, l, a)
    content_mask_origin: [f32; 2], // offset 32
    content_mask_size: [f32; 2],   // offset 40
} // stride  48

// Stride expected by `array<GpuPathVertex>` in paths.wgsl's storage buffer.
const _: () = assert!(std::mem::size_of::<GpuPathVertex>() == 48);

struct UnderlinesData {
    globals: GlobalParams,
}

struct MonoSpritesData {
    globals: GlobalParams,
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    t_sprite: wgpu::TextureView,
    s_sprite: wgpu::Sampler,
}

struct PolySpritesData {
    globals: GlobalParams,
    t_sprite: wgpu::TextureView,
    s_sprite: wgpu::Sampler,
}

struct SurfacesData {
    globals: GlobalParams,
    surface_params: SurfaceParams,
    t_y: wgpu::TextureView,
    t_cb_cr: wgpu::TextureView,
    s_texture: wgpu::Sampler,
}

struct PathSprite {
    bounds: geometry::Bounds<f32>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PathRasterizationVertex {
    xy_position: geometry::Point<ScaledPixels>,
    st_position: geometry::Point<f32>,
    color: color::Background,
    bounds: geometry::Bounds<f32>,
}

impl PathRasterizationVertex {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 10] = &{
        let color_vertex_attributes = map_attributes(
            color::Background::VERTEX_ATTRIBUTES,
            2,
            std::mem::offset_of!(PathRasterizationVertex, color) as wgpu::BufferAddress,
        );

        let bounds_vertex_attributes = map_attributes(
            Bounds::VERTEX_ATTRIBUTES,
            8,
            std::mem::offset_of!(PathRasterizationVertex, bounds) as wgpu::BufferAddress,
        );

        [
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(PathRasterizationVertex, xy_position)
                    as wgpu::BufferAddress,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(PathRasterizationVertex, st_position)
                    as wgpu::BufferAddress,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            },
            color_vertex_attributes[0],
            color_vertex_attributes[1],
            color_vertex_attributes[2],
            color_vertex_attributes[3],
            color_vertex_attributes[4],
            color_vertex_attributes[5],
            bounds_vertex_attributes[0],
            bounds_vertex_attributes[1],
        ]
    };

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PathRasterizationVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: Self::VERTEX_ATTRIBUTES,
        }
    }
}

impl AtlasTextureId {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 2] = &{
        [
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(AtlasTextureId, index) as wgpu::BufferAddress,
                format: wgpu::VertexFormat::Uint32,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(AtlasTextureId, kind) as wgpu::BufferAddress,
                format: wgpu::VertexFormat::Uint32,
                shader_location: 1,
            },
        ]
    };
}

#[repr(C)]
struct AtlasBounds {
    origin: [i32; 2],
    size: [i32; 2],
}

impl AtlasBounds {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 2] = &{
        [
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(AtlasBounds, origin) as wgpu::BufferAddress,
                format: wgpu::VertexFormat::Sint32x2,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(AtlasBounds, size) as wgpu::BufferAddress,
                format: wgpu::VertexFormat::Sint32x2,
                shader_location: 1,
            },
        ]
    };
}

impl AtlasTile {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 6] = &{
        let texture_id_vertex_attributes = map_attributes(
            AtlasTextureId::VERTEX_ATTRIBUTES,
            0,
            std::mem::offset_of!(AtlasTile, texture_id) as wgpu::BufferAddress,
        );

        let bounds_vertex_attributes = map_attributes(
            AtlasBounds::VERTEX_ATTRIBUTES,
            4,
            std::mem::offset_of!(AtlasTile, bounds) as wgpu::BufferAddress,
        );

        [
            texture_id_vertex_attributes[0],
            texture_id_vertex_attributes[1],
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(AtlasTile, tile_id) as wgpu::BufferAddress,
                format: wgpu::VertexFormat::Uint32,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(AtlasTile, padding) as wgpu::BufferAddress,
                format: wgpu::VertexFormat::Uint32,
                shader_location: 3,
            },
            bounds_vertex_attributes[0],
            bounds_vertex_attributes[1],
        ]
    };
}

impl TransformationMatrix {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 2] = &{
        [
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(TransformationMatrix, rotation_scale)
                    as wgpu::BufferAddress,
                format: wgpu::VertexFormat::Float32x4,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(TransformationMatrix, translation)
                    as wgpu::BufferAddress,
                format: wgpu::VertexFormat::Float32x2,
                shader_location: 1,
            },
        ]
    };
}

impl MonochromeSprite {
    const VERTEX_ATTRIBUTES: &'static [wgpu::VertexAttribute; 21] = &{
        let bounds_vertex_attributes = map_attributes(
            Bounds::VERTEX_ATTRIBUTES,
            2,
            std::mem::offset_of!(MonochromeSprite, bounds) as wgpu::BufferAddress,
        );

        let content_mask_vertex_attributes = map_attributes(
            Bounds::VERTEX_ATTRIBUTES,
            4,
            std::mem::offset_of!(MonochromeSprite, content_mask) as wgpu::BufferAddress,
        );

        let text_color_vertex_attributes = map_attributes(
            color::TextColor::VERTEX_ATTRIBUTES,
            6,
            std::mem::offset_of!(MonochromeSprite, text_color) as wgpu::BufferAddress,
        );

        let tile_vertex_attributes = map_attributes(
            AtlasTile::VERTEX_ATTRIBUTES,
            8,
            std::mem::offset_of!(MonochromeSprite, tile) as wgpu::BufferAddress,
        );

        let transformation_matrix_vertex_attributes = map_attributes(
            TransformationMatrix::VERTEX_ATTRIBUTES,
            14,
            std::mem::offset_of!(MonochromeSprite, transformation) as wgpu::BufferAddress,
        );

        [
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(MonochromeSprite, order) as wgpu::BufferAddress,
                format: wgpu::VertexFormat::Uint32,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(MonochromeSprite, pad) as wgpu::BufferAddress,
                format: wgpu::VertexFormat::Uint32,
                shader_location: 1,
            },
            bounds_vertex_attributes[0],
            bounds_vertex_attributes[1],
            content_mask_vertex_attributes[0],
            content_mask_vertex_attributes[1],
            text_color_vertex_attributes[0],
            text_color_vertex_attributes[1],
            text_color_vertex_attributes[2],
            text_color_vertex_attributes[3],
            text_color_vertex_attributes[4],
            text_color_vertex_attributes[5],
            text_color_vertex_attributes[6],
            tile_vertex_attributes[0],
            tile_vertex_attributes[1],
            tile_vertex_attributes[2],
            tile_vertex_attributes[3],
            tile_vertex_attributes[4],
            tile_vertex_attributes[5],
            transformation_matrix_vertex_attributes[0],
            transformation_matrix_vertex_attributes[1],
        ]
    };
}

#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct ColorAdjustments {
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    _padding: [f32; 3],
}

// Size of the `ColorAdjustments` uniform in mono_sprites.wgsl, rounded up to
// the 16-byte uniform binding alignment (the `_padding` field supplies it).
const _: () = assert!(std::mem::size_of::<ColorAdjustments>() == 32);

struct WgpuPipelines {
    color_targets: Vec<Option<wgpu::ColorTargetState>>,

    quads_bind_group_layout: wgpu::BindGroupLayout,
    shadows_bind_group_layout: wgpu::BindGroupLayout,
    backdrop_filters_bind_group_layout: wgpu::BindGroupLayout,
    backdrop_texture_bind_group_layout: wgpu::BindGroupLayout,
    underlines_bind_group_layout: wgpu::BindGroupLayout,
    sprites_bind_group_layout: wgpu::BindGroupLayout,
    mono_sprites_bind_group_layout: wgpu::BindGroupLayout,
    poly_sprites_bind_group_layout: wgpu::BindGroupLayout,
    surfaces_bind_group_layout: wgpu::BindGroupLayout,
    paths_bind_group_layout: wgpu::BindGroupLayout,
    /// Per-layer translate, bound at the highest position of every pipeline
    /// that can draw slab content. Dynamic-offset so one small uniform serves
    /// all layers; slot 0 is permanently zero for legacy draws.
    layer_transform_bind_group_layout: wgpu::BindGroupLayout,

    globals_bind_group: wgpu::BindGroup,
    color_adjustments_bind_group: wgpu::BindGroup,

    quads_pipeline: wgpu::RenderPipeline,
    shadows_pipeline: wgpu::RenderPipeline,
    backdrop_filters_pipeline: wgpu::RenderPipeline,
    underlines_pipeline: wgpu::RenderPipeline,
    mono_sprites_pipeline: wgpu::RenderPipeline,
    poly_sprites_pipeline: wgpu::RenderPipeline,
    surfaces_pipeline: wgpu::RenderPipeline,
    paths_pipeline: wgpu::RenderPipeline,
}

impl WgpuPipelines {
    pub fn new(
        context: &WgpuContext,
        surface_configuration: &wgpu::SurfaceConfiguration,
        _path_sample_count: u32,
    ) -> Self {
        let quads_shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("quads_shader"),
                source: wgpu::ShaderSource::Wgsl(slab_shader_source(
                    "quads",
                    2,
                    include_str!("shaders/quads.wgsl"),
                )),
            });

        let shadows_shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shadows_shader"),
                source: wgpu::ShaderSource::Wgsl(slab_shader_source(
                    "shadows",
                    2,
                    include_str!("shaders/shadows.wgsl"),
                )),
            });

        let backdrop_filter_shader =
            context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("backdrop_filter_shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/backdrop_blur.wgsl").into(),
                    ),
                });

        let underlines_shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("underlines_shader"),
                source: wgpu::ShaderSource::Wgsl(slab_shader_source("underlines", 2, include_str!("shaders/underlines.wgsl"))),
            });

        let mono_sprite_shader =
            context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("mono_sprites shader"),
                    source: wgpu::ShaderSource::Wgsl(slab_shader_source(
                        "mono_sprites",
                        4,
                        include_str!("shaders/mono_sprites.wgsl"),
                    )),
                });

        let poly_sprite_shader =
            context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("poly_sprites shader"),
                    source: wgpu::ShaderSource::Wgsl(slab_shader_source(
                        "poly_sprites",
                        3,
                        include_str!("shaders/poly_sprites.wgsl"),
                    )),
                });

        let blend_mode = match surface_configuration.alpha_mode {
            wgpu::CompositeAlphaMode::PreMultiplied => {
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
            }
            _ => wgpu::BlendState::ALPHA_BLENDING,
        };

        let color_targets = &[Some(wgpu::ColorTargetState {
            format: surface_configuration.format,
            blend: Some(blend_mode),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let globals_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("globals"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let color_adjustments_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("color_adjustments_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let sprites_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("sprite_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let layer_transform_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("layer_transform_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: Some(
                                std::num::NonZeroU64::new(std::mem::size_of::<GpuLayerTransform>() as u64)
                                    .expect("non-zero transform size"),
                            ),
                        },
                        count: None,
                    }],
                });

        let quads_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("quads_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let quads_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("quads_pipeline_layout"),
                    bind_group_layouts: &[
                        Some(&globals_bind_group_layout),
                        Some(&quads_bind_group_layout),
                        Some(&layer_transform_bind_group_layout),
                    ],
                    immediate_size: 0,
                });

        let shadows_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("shadows_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let shadows_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("shadows_pipeline_layout"),
                    bind_group_layouts: &[
                        Some(&globals_bind_group_layout),
                        Some(&shadows_bind_group_layout),
                        Some(&layer_transform_bind_group_layout),
                    ],
                    immediate_size: 0,
                });

        let backdrop_filters_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("backdrop_filters_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let backdrop_texture_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("backdrop_texture_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let backdrop_filters_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("backdrop_filters_pipeline_layout"),
                    bind_group_layouts: &[
                        Some(&globals_bind_group_layout),
                        Some(&backdrop_filters_bind_group_layout),
                        Some(&backdrop_texture_bind_group_layout),
                    ],
                    immediate_size: 0,
                });

        let underlines_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("underlines_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let underlines_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("underlines_pipeline_layout"),
                    bind_group_layouts: &[
                        Some(&globals_bind_group_layout),
                        Some(&underlines_bind_group_layout),
                        Some(&layer_transform_bind_group_layout),
                    ],
                    immediate_size: 0,
                });

        let mono_sprites_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Mono sprites bind group layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let mono_sprites_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Mono sprites pipeline layout"),
                    bind_group_layouts: &[
                        Some(&globals_bind_group_layout),
                        Some(&color_adjustments_bind_group_layout),
                        Some(&sprites_bind_group_layout),
                        Some(&mono_sprites_bind_group_layout),
                        Some(&layer_transform_bind_group_layout),
                    ],
                    immediate_size: 0,
                });

        let poly_sprites_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Poly sprites bind group layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let poly_sprites_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Poly sprites pipeline layout"),
                    bind_group_layouts: &[
                        Some(&globals_bind_group_layout),
                        Some(&sprites_bind_group_layout),
                        Some(&poly_sprites_bind_group_layout),
                        Some(&layer_transform_bind_group_layout),
                    ],
                    immediate_size: 0,
                });

        let surfaces_shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("surfaces_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/surfaces.wgsl").into()),
            });

        let surfaces_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("surfaces_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let surfaces_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("surfaces_pipeline_layout"),
                    bind_group_layouts: &[
                        Some(&globals_bind_group_layout),
                        Some(&surfaces_bind_group_layout),
                    ],
                    immediate_size: 0,
                });

        let globals_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("globals_bind_group"),
                layout: &globals_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &context.globals_buffer,
                        offset: 0,
                        size: None,
                    }),
                }],
            });

        let color_adjustments_bind_group =
            context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("color_adjustments_bind_group"),
                    layout: &color_adjustments_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &context.color_adjustments_buffer,
                            offset: 0,
                            size: None,
                        }),
                    }],
                });

        // ---- Paths pipeline ------------------------------------------------
        let paths_shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("paths_shader"),
                source: wgpu::ShaderSource::Wgsl(slab_shader_source(
                    "paths",
                    2,
                    include_str!("shaders/paths.wgsl"),
                )),
            });

        let paths_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("paths_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let paths_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("paths_pipeline_layout"),
                    bind_group_layouts: &[
                        Some(&globals_bind_group_layout),
                        Some(&paths_bind_group_layout),
                        Some(&layer_transform_bind_group_layout),
                    ],
                    immediate_size: 0,
                });
        // --------------------------------------------------------------------

        Self {
            color_targets: color_targets.to_vec(),

            quads_bind_group_layout,
            shadows_bind_group_layout,
            backdrop_filters_bind_group_layout,
            backdrop_texture_bind_group_layout,
            underlines_bind_group_layout,
            mono_sprites_bind_group_layout,
            sprites_bind_group_layout,
            poly_sprites_bind_group_layout,
            paths_bind_group_layout,
            layer_transform_bind_group_layout,

            globals_bind_group,
            color_adjustments_bind_group,

            quads_pipeline: context.device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("quads"),
                    layout: Some(&quads_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &quads_shader,
                        entry_point: Some("vs_quad"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &quads_shader,
                        entry_point: Some("fs_quad"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: color_targets,
                    }),
                    multiview_mask: None,
                    cache: None,
                },
            ),

            shadows_pipeline: context.device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("shadows"),
                    layout: Some(&shadows_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shadows_shader,
                        entry_point: Some("vs_shadow"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &shadows_shader,
                        entry_point: Some("fs_shadow"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: color_targets,
                    }),
                    multiview_mask: None,
                    cache: None,
                },
            ),

            backdrop_filters_pipeline: context.device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("backdrop_filters"),
                    layout: Some(&backdrop_filters_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &backdrop_filter_shader,
                        entry_point: Some("vs_backdrop_filter"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &backdrop_filter_shader,
                        entry_point: Some("fs_backdrop_filter"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: color_targets,
                    }),
                    multiview_mask: None,
                    cache: None,
                },
            ),

            underlines_pipeline: context.device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("underlines"),
                    layout: Some(&underlines_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &underlines_shader,
                        entry_point: Some("vs_underline"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &underlines_shader,
                        entry_point: Some("fs_underline"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: color_targets,
                    }),
                    multiview_mask: None,
                    cache: None,
                },
            ),

            mono_sprites_pipeline: context.device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("mono_sprites"),
                    layout: Some(&mono_sprites_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &mono_sprite_shader,
                        entry_point: Some("vs_mono_sprite"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    fragment: Some(wgpu::FragmentState {
                        module: &mono_sprite_shader,
                        entry_point: Some("fs_mono_sprite"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: color_targets,
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                },
            ),

            poly_sprites_pipeline: context.device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("poly_sprites"),
                    layout: Some(&poly_sprites_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &poly_sprite_shader,
                        entry_point: Some("vs_poly_sprite"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    fragment: Some(wgpu::FragmentState {
                        module: &poly_sprite_shader,
                        entry_point: Some("fs_poly_sprite"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: color_targets,
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                },
            ),

            surfaces_bind_group_layout,

            surfaces_pipeline: context.device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("surfaces"),
                    layout: Some(&surfaces_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &surfaces_shader,
                        entry_point: Some("vs_surface"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    fragment: Some(wgpu::FragmentState {
                        module: &surfaces_shader,
                        entry_point: Some("fs_surface"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: color_targets,
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                },
            ),

            paths_pipeline: context.device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("paths"),
                    layout: Some(&paths_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &paths_shader,
                        entry_point: Some("vs_path"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &paths_shader,
                        entry_point: Some("fs_path"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: color_targets,
                    }),
                    multiview_mask: None,
                    cache: None,
                },
            ),
        }
    }
}

struct RenderingParameters {
    path_sample_count: u32,
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
}

/// Bind state for one frame's slab draws, created only when the frame
/// actually carries spans.
struct SlabDrawGroups {
    quads: wgpu::BindGroup,
    shadows: wgpu::BindGroup,
    paths_vertices: wgpu::BindGroup,
    underlines: wgpu::BindGroup,
    mono_sprites: wgpu::BindGroup,
    poly_sprites: wgpu::BindGroup,
    layer_transform: wgpu::BindGroup,
    /// Keyed by `(index, kind)`: `AtlasTextureId` carries no `Hash`, and the
    /// pair is what actually identifies a live page binding.
    sprite_textures: FxHashMap<(u32, crate::AtlasTextureKind), wgpu::BindGroup>,
}

/// One merged stretch of a layer's slab stream awaiting its draw.
#[derive(Clone, Copy)]
struct SlabPendingRun {
    kind: SlabKind,
    texture_id: Option<AtlasTextureId>,
    start: u32,
    count: u32,
}

/// Which render pipeline is currently bound, as a semantic identity the bind
/// tracker can compare without touching wgpu resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DrawPipelineId {
    Quads,
    Shadows,
    Paths,
    Underlines,
    MonoSprites,
    PolySprites,
}

/// Which legacy fixed buffer a bind group wraps, per kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LegacyBuffer {
    Quads,
    Shadows,
    Underlines,
    MonoSprites,
    PolySprites,
    PathVertices,
}

/// Semantic identity of one bindable resource at one bind-group slot.
///
/// Two draws whose ids agree at every slot bind byte-identical GPU state, so
/// the second set is skippable without any pixel effect. Resources this
/// module does not model (per-surface groups, filter composites) are never
/// given an id: those draw paths reset the tracker instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundGroupId {
    Globals,
    ColorAdjustments,
    /// The layer transform uniform at a specific dynamic offset. Legacy
    /// draws always use offset 0 (the identity slot); slab draws use
    /// `slot * stride` for their layer.
    LayerTransform(u32),
    LegacyBuffer(LegacyBuffer),
    SlabStorage(SlabKind),
    SpriteTexture(u32, crate::AtlasTextureKind),
}

/// Upper bound on bind-group slots any pipeline here uses (mono sprites: 5).
const PASS_BIND_SLOTS: usize = 5;

/// What the current render pass has bound, so redundant
/// `set_pipeline`/`set_bind_group` calls can be skipped.
///
/// wgpu offers no way to query a pass's bound state, and driver-side state
/// churn is exactly what this tracks: consecutive same-kind runs (and split
/// legacy batches) re-issue identical binds today. Ids are semantic rather
/// than pointer-based, which keeps skipping sound even when equal-content
/// bind groups are distinct objects. Anything unmodeled must call [`Self::reset`]
/// before tracked draws resume.
#[derive(Default)]
struct PassBindState {
    pipeline: Option<DrawPipelineId>,
    groups: [Option<BoundGroupId>; PASS_BIND_SLOTS],
}

impl PassBindState {
    fn reset(&mut self) {
        self.pipeline = None;
        self.groups = [None; PASS_BIND_SLOTS];
    }

    fn set_pipeline(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        id: DrawPipelineId,
        pipeline: &wgpu::RenderPipeline,
    ) {
        if self.pipeline == Some(id) {
            return;
        }
        pass.set_pipeline(pipeline);
        self.pipeline = Some(id);
    }

    fn set_bind_group(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        index: u32,
        id: BoundGroupId,
        group: &wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    ) {
        let slot = index as usize;
        if slot < PASS_BIND_SLOTS && self.groups[slot] == Some(id) {
            return;
        }
        pass.set_bind_group(index, group, offsets);
        if slot < PASS_BIND_SLOTS {
            self.groups[slot] = Some(id);
        }
    }
}

/// One merged slab stretch opened by a span that may stay open across span
/// boundaries until a non-continuing draw forces its flush.
///
/// Merging requires more than instance contiguity: both stretches must draw
/// through the same transform uniform offset, i.e. belong to the same layer.
/// Different layers occupy different transform slots, so their instances can
/// sit adjacent in a kind buffer yet still need separate draws — cross-layer
/// merging is impossible without changing pixels (or relocating resident
/// bytes to co-locate layers, a buffer reshuffle this deliberately avoids).
struct OpenSlabRun {
    key: LayerKey,
    slabs: crate::platform::cross::slab::LayerSlabs,
    transform_slot: u32,
    kind: SlabKind,
    texture_id: Option<AtlasTextureId>,
    start: u32,
    count: u32,
}

impl OpenSlabRun {
    /// Whether `run` continues this stretch: same layer, same kind, same
    /// texture, and exactly contiguous in the layer-wide instance stream.
    fn accepts(
        &self,
        key: LayerKey,
        slabs: &crate::platform::cross::slab::LayerSlabs,
        run: &crate::scene::SlabRun,
    ) -> bool {
        self.key == key
            && &self.slabs == slabs
            && self.kind == run.kind
            && self.texture_id == run.texture_id
            && self.start + self.count == run.start
    }

    fn as_pending(&self) -> SlabPendingRun {
        SlabPendingRun {
            kind: self.kind,
            texture_id: self.texture_id,
            start: self.start,
            count: self.count,
        }
    }
}

/// Frame-to-frame cache of the slab bind state: the six per-kind storage
/// groups plus the transform-uniform group are rebuilt only when
/// [`SlabGpuBuffers`] recreates their buffer, and atlas-page groups refresh
/// only when the referenced-page set changes. On a Clean-only frame this
/// costs a few handle clones instead of eight `create_bind_group` calls.
struct SlabGroupCache {
    kind_groups: [Option<wgpu::BindGroup>; SlabKind::COUNT],
    transforms: Option<wgpu::BindGroup>,
    /// The canonical (sorted) page set `sprite_textures` was built from.
    pages: Vec<(u32, crate::AtlasTextureKind)>,
    page_scratch: Vec<(u32, crate::AtlasTextureKind)>,
    sprite_textures: FxHashMap<(u32, crate::AtlasTextureKind), wgpu::BindGroup>,
    #[cfg(test)]
    creations: u64,
}

impl Default for SlabGroupCache {
    fn default() -> Self {
        SlabGroupCache {
            kind_groups: std::array::from_fn(|_| None),
            transforms: None,
            pages: Vec::new(),
            page_scratch: Vec::new(),
            sprite_textures: FxHashMap::default(),
            #[cfg(test)]
            creations: 0,
        }
    }
}

impl SlabGroupCache {
    #[cfg(test)]
    fn creation_count(&self) -> u64 {
        self.creations
    }

    fn invalidate_kind(&mut self, kind: SlabKind) {
        self.kind_groups[kind.index()] = None;
    }

    fn invalidate_transforms(&mut self) {
        self.transforms = None;
    }

    fn kind_layout(pipelines: &WgpuPipelines, kind: SlabKind) -> &wgpu::BindGroupLayout {
        match kind {
            SlabKind::Quads => &pipelines.quads_bind_group_layout,
            SlabKind::Shadows => &pipelines.shadows_bind_group_layout,
            SlabKind::Paths => &pipelines.paths_bind_group_layout,
            SlabKind::Underlines => &pipelines.underlines_bind_group_layout,
            SlabKind::MonoSprites => &pipelines.mono_sprites_bind_group_layout,
            SlabKind::PolySprites => &pipelines.poly_sprites_bind_group_layout,
        }
    }

    fn ensure_kind_group(
        &mut self,
        device: &wgpu::Device,
        pipelines: &WgpuPipelines,
        buffers: &slab_gpu::SlabGpuBuffers,
        kind: SlabKind,
    ) {
        let index = kind.index();
        if self.kind_groups[index].is_some() {
            return;
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slab_kind_bind_group"),
            layout: Self::kind_layout(pipelines, kind),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buffers.kind_buffer(kind),
                    offset: 0,
                    size: None,
                }),
            }],
        });
        self.kind_groups[index] = Some(group);
        #[cfg(test)]
        {
            self.creations += 1;
        }
    }

    fn ensure_transforms_group(
        &mut self,
        device: &wgpu::Device,
        pipelines: &WgpuPipelines,
        buffers: &slab_gpu::SlabGpuBuffers,
    ) {
        if self.transforms.is_some() {
            return;
        }        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_transform_bind_group"),
            layout: &pipelines.layer_transform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buffers.transforms_buffer(),
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(
                        std::mem::size_of::<GpuLayerTransform>() as u64,
                    )
                    .expect("non-zero transform size")),
                }),
            }],
        });
        self.transforms = Some(group);
        #[cfg(test)]
        {
            self.creations += 1;
        }
    }

    /// The cached transform-uniform bind group, recreated only after the
    /// uniform buffer was. Legacy draws share this group with slab draws:
    /// both bind the same uniform, selecting slots via dynamic offsets.
    fn transforms_group(
        &mut self,
        device: &wgpu::Device,
        pipelines: &WgpuPipelines,
        buffers: &slab_gpu::SlabGpuBuffers,
    ) -> wgpu::BindGroup {
        self.ensure_transforms_group(device, pipelines, buffers);
        self.transforms.as_ref().expect("just ensured").clone()
    }

    /// Rebuild the page map only when this frame's referenced-page set differs
    /// from the cached one. Pages whose layers were poisoned by eviction stay
    /// in the map but are never bound: poisoned layers skip their draws before
    /// the texture check runs.
    fn sync_sprite_pages(
        &mut self,
        device: &wgpu::Device,
        pipelines: &WgpuPipelines,
        atlas: &WgpuAtlas,
        atlas_sampler: &wgpu::Sampler,
        scene: &Scene,
    ) {
        self.page_scratch.clear();
        for span in &scene.layer_slab_spans {
            for run in &span.runs {
                if let Some(texture_id) = run.texture_id {
                    let key = (texture_id.index, texture_id.kind);
                    if !self.page_scratch.contains(&key) {
                        self.page_scratch.push(key);
                    }
                }
            }
        }
        // Sorted so bind-group creation order stays deterministic under fuzzing.
        self.page_scratch.sort_by_key(|&(index, kind)| (index, kind as u8));
        if self.pages == self.page_scratch {
            return;
        }
        self.sprite_textures.clear();
        #[cfg(test)]
        let rebuilt = self.page_scratch.len() as u64;
        for &(texture_index, texture_kind) in &self.page_scratch {
            let tex_info = atlas.get_texture_info(AtlasTextureId {
                index: texture_index,
                kind: texture_kind,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("slab_sprite_texture_bind_group"),
                layout: &pipelines.sprites_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&tex_info.raw_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(atlas_sampler),
                    },
                ],
            });
            self.sprite_textures.insert((texture_index, texture_kind), group);
        }
        std::mem::swap(&mut self.pages, &mut self.page_scratch);
        #[cfg(test)]
        {
            self.creations += rebuilt;
        }
    }

    /// The frame's slab bind state, cloned out of the cache.
    fn frame_groups(
        &mut self,
        device: &wgpu::Device,
        pipelines: &WgpuPipelines,
        buffers: &slab_gpu::SlabGpuBuffers,
        atlas: &WgpuAtlas,
        atlas_sampler: &wgpu::Sampler,
        scene: &Scene,
    ) -> SlabDrawGroups {
        for kind in SlabKind::ALL {
            self.ensure_kind_group(device, pipelines, buffers, kind);
        }
        self.ensure_transforms_group(device, pipelines, buffers);
        self.sync_sprite_pages(device, pipelines, atlas, atlas_sampler, scene);
        let [quads, shadows, paths_vertices, underlines, mono_sprites, poly_sprites] =
            &self.kind_groups;
        SlabDrawGroups {
            quads: quads.as_ref().expect("kind group ensured above").clone(),
            shadows: shadows.as_ref().expect("kind group ensured above").clone(),
            paths_vertices: paths_vertices
                .as_ref()
                .expect("kind group ensured above")
                .clone(),
            underlines: underlines
                .as_ref()
                .expect("kind group ensured above")
                .clone(),
            mono_sprites: mono_sprites
                .as_ref()
                .expect("kind group ensured above")
                .clone(),
            poly_sprites: poly_sprites
                .as_ref()
                .expect("kind group ensured above")
                .clone(),
            layer_transform: self.transforms.as_ref().expect("transform group ensured above").clone(),
            sprite_textures: self.sprite_textures.clone(),
        }
    }
}

/// Bind state for one frame's slab draws, created only when the frame
/// actually carries slots. Free-standing so the GPU-tier tests drive the
/// exact production construction; the frame path uses [`SlabGroupCache`].
#[cfg(test)]
fn build_slab_draw_groups(
    device: &wgpu::Device,
    pipelines: &WgpuPipelines,
    buffers: &slab_gpu::SlabGpuBuffers,
    atlas: &WgpuAtlas,
    atlas_sampler: &wgpu::Sampler,
    layer_transform_bind_group: &wgpu::BindGroup,
    scene: &Scene,
) -> SlabDrawGroups {
    let buffer_group = |label: &'static str,
                        layout: &wgpu::BindGroupLayout,
                        buffer: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        })
    };

    let mut sprite_textures: FxHashMap<(u32, crate::AtlasTextureKind), wgpu::BindGroup> =
        FxHashMap::default();
    let mut textures_this_frame: Vec<(u32, crate::AtlasTextureKind)> = Vec::new();
    for span in &scene.layer_slab_spans {
        for run in &span.runs {
            if let Some(texture_id) = run.texture_id {
                let key = (texture_id.index, texture_id.kind);
                if !textures_this_frame.contains(&key) {
                    textures_this_frame.push(key);
                }
            }
        }
    }
    // Sorted so bind-group creation order is deterministic under fuzzing.
    textures_this_frame.sort_by_key(|&(index, kind)| (index, kind as u8));
    for (texture_index, texture_kind) in textures_this_frame {
        let texture_id = AtlasTextureId {
            index: texture_index,
            kind: texture_kind,
        };
        let tex_info = atlas.get_texture_info(texture_id);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slab_sprite_texture_bind_group"),
            layout: &pipelines.sprites_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_info.raw_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(atlas_sampler),
                },
            ],
        });
        sprite_textures.insert((texture_index, texture_kind), group);
    }

    SlabDrawGroups {
        quads: buffer_group(
            "slab_quads_bind_group",
            &pipelines.quads_bind_group_layout,
            buffers.kind_buffer(SlabKind::Quads),
        ),
        shadows: buffer_group(
            "slab_shadows_bind_group",
            &pipelines.shadows_bind_group_layout,
            buffers.kind_buffer(SlabKind::Shadows),
        ),
        paths_vertices: buffer_group(
            "slab_paths_vertices_bind_group",
            &pipelines.paths_bind_group_layout,
            buffers.kind_buffer(SlabKind::Paths),
        ),
        underlines: buffer_group(
            "slab_underlines_bind_group",
            &pipelines.underlines_bind_group_layout,
            buffers.kind_buffer(SlabKind::Underlines),
        ),
        mono_sprites: buffer_group(
            "slab_mono_sprites_bind_group",
            &pipelines.mono_sprites_bind_group_layout,
            buffers.kind_buffer(SlabKind::MonoSprites),
        ),
        poly_sprites: buffer_group(
            "slab_poly_sprites_bind_group",
            &pipelines.poly_sprites_bind_group_layout,
            buffers.kind_buffer(SlabKind::PolySprites),
        ),
        layer_transform: layer_transform_bind_group.clone(),
        sprite_textures,
    }
}

/// Instances a legacy primitive batch would draw. Zero-instance batches
/// (the empty split halves `FrameBatchIterator` queues around spans) draw
/// nothing, so they must not break an open slab stretch's adjacency.
fn primitive_batch_instance_count(batch: &PrimitiveBatch<'_>) -> u32 {
    match batch {
        PrimitiveBatch::Quads(quads) => quads.len() as u32,
        PrimitiveBatch::Shadows(shadows) => shadows.len() as u32,
        PrimitiveBatch::Paths(paths) => paths.iter().map(|p| p.vertices.len() as u32).sum(),
        PrimitiveBatch::Underlines(underlines) => underlines.len() as u32,
        PrimitiveBatch::MonochromeSprites { sprites, .. } => sprites.len() as u32,
        PrimitiveBatch::PolychromeSprites { sprites, .. } => sprites.len() as u32,
        PrimitiveBatch::Surfaces(surfaces) => surfaces.len() as u32,
        PrimitiveBatch::BackdropFilters(backdrop_filters) => backdrop_filters.len() as u32,
        // One marker, one (degenerate) draw.
        PrimitiveBatch::FilterBoundary(_) => 1,
    }
}

/// One shared pass over the scene's spans, grouping every referenced atlas
/// page by owning layer.
///
/// Replaces the per-synced-layer rescan of all spans (O(spans²) with the
/// filter inside the sync loop); residency bookkeeping consumes the result
/// without changing what is recorded per layer.
fn collect_referenced_pages_by_layer(
    scene: &Scene,
) -> FxHashMap<LayerKey, Vec<(u32, crate::AtlasTextureKind)>> {
    let mut pages: FxHashMap<LayerKey, Vec<(u32, crate::AtlasTextureKind)>> =
        FxHashMap::default();
    for span in &scene.layer_slab_spans {
        for run in &span.runs {
            if let Some(texture_id) = run.texture_id {
                pages
                    .entry(span.key)
                    .or_default()
                    .push((texture_id.index, texture_id.kind));
            }
        }
    }
    pages
}

fn append_packed_kind_bytes(
    scratch: &mut Vec<u8>,
    kind: SlabKind,
    packed: &crate::scene_pack::PackedLayer,
) {
    match kind {
        SlabKind::Quads => scratch.extend_from_slice(bytemuck::cast_slice(&packed.quads)),
        SlabKind::Shadows => scratch.extend_from_slice(bytemuck::cast_slice(&packed.shadows)),
        // Path slabs hold the flattened GpuPathVertex stream (color and mask
        // baked per vertex), exactly what the legacy upload builds.
        SlabKind::Paths => {
            for path in &packed.paths {
                let color = path.color.solid;
                let cm = &path.content_mask.bounds;
                let cm_origin = [cm.origin.x.0, cm.origin.y.0];
                let cm_size = [cm.size.width.0, cm.size.height.0];
                for vertex in &path.vertices {
                    scratch.extend_from_slice(bytemuck::bytes_of(&GpuPathVertex {
                        xy_position: [vertex.xy_position.x.0, vertex.xy_position.y.0],
                        st_position: [vertex.st_position.x, vertex.st_position.y],
                        hsla: [color.h, color.s, color.l, color.a],
                        content_mask_origin: cm_origin,
                        content_mask_size: cm_size,
                    }));
                }
            }
        }
        SlabKind::Underlines => {
            scratch.extend_from_slice(bytemuck::cast_slice(&packed.underlines))
        }
        SlabKind::MonoSprites => {
            scratch.extend_from_slice(bytemuck::cast_slice(&packed.mono_sprites))
        }
        SlabKind::PolySprites => {
            scratch.extend_from_slice(bytemuck::cast_slice(&packed.poly_sprites))
        }
    }
}

impl RenderingParameters {
    fn from_env() -> Self {
        use std::env;

        let path_sample_count = env::var("ZED_PATH_SAMPLE_COUNT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let gamma = env::var("ZED_FONTS_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.8_f32)
            .clamp(1.0, 2.2);
        let gamma_ratios = crate::platform::get_gamma_correction_ratios(gamma);
        let grayscale_enhanced_contrast = env::var("ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0_f32)
            .max(0.0);

        Self {
            path_sample_count,
            gamma_ratios,
            grayscale_enhanced_contrast,
        }
    }
}

/// Cached bounds information for fast surface blitting
#[derive(Clone, Debug)]
struct SurfaceBoundsEntry {
    /// Screen-space bounds where the surface should be rendered
    screen_bounds: geometry::Bounds<Pixels>,
    /// Content mask for clipping
    content_mask: geometry::Bounds<Pixels>,
    /// Layout version when these bounds were computed (for staleness detection)
    layout_version: u64,
}

/// Maximum nesting depth supported for CSS-style content `filter` groups
/// (`with_filter_layer`). Groups nested deeper than this are painted inline,
/// unisolated and unblurred, rather than allocating unbounded offscreen textures.
const MAX_FILTER_DEPTH: usize = 4;

/// How many frames a layer texture may go unreferenced before the cache
/// drops it and posts a re-record request for its layer (#96). Generous, so
/// a briefly-scrolled-away buffer never thrashes.
const LAYER_TEXTURE_IDLE_FRAMES: u64 = 240;

/// One texture-retained layer's persistent offscreen texture (#96).
struct LayerTextureEntry {
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    /// The owning layer, for re-record requests when the entry dies.
    key: crate::LayerKey,
    /// The content generation baked in; compared against span tokens.
    content_token: u64,
    /// The buffer extent the texture was created at, in scaled window pixels.
    texture_bounds: crate::Bounds<crate::ScaledPixels>,
    last_used_frame: u64,
}

/// Allocates the pool of full-surface-sized offscreen textures that
/// content-filter groups render into, one per supported nesting depth.
fn create_filter_group_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (Vec<wgpu::Texture>, Vec<wgpu::TextureView>) {
    (0..MAX_FILTER_DEPTH)
        .map(|_| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("filter_group_texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        })
        .unzip()
}

pub struct WgpuRenderer {
    context: Arc<WgpuContext>,
    surface: ManuallyDrop<wgpu::Surface<'static>>,
    surface_configuration: wgpu::SurfaceConfiguration,
    atlas_sampler: wgpu::Sampler,
    surface_sampler: wgpu::Sampler,
    atlas: Arc<WgpuAtlas>,
    pipelines: WgpuPipelines,
    rendering_parameters: RenderingParameters,

    // cache bind groups for each double-buffered surface (index 0/1)
    surface_bind_groups:
        Mutex<HashMap<crate::platform::cross::surface_registry::SurfaceId, [wgpu::BindGroup; 2]>>,

    // Persistent framebuffer for browser-canvas-style blitting
    persistent_framebuffer: Option<wgpu::Texture>,
    persistent_framebuffer_view: Option<wgpu::TextureView>,

    // Backdrop blur texture for capturing framebuffer content
    backdrop_blur_texture: Option<wgpu::Texture>,
    backdrop_blur_texture_view: Option<wgpu::TextureView>,
    backdrop_blur_sampler: wgpu::Sampler,

    // Pool of offscreen textures that content-filter groups (`with_filter_layer`)
    // render into so their content can be blurred and composited as a unit.
    // Indexed by nesting depth, sized to `MAX_FILTER_DEPTH`.
    group_textures: Vec<wgpu::Texture>,
    group_views: Vec<wgpu::TextureView>,

    // Bounds cache for fast surface blitting without compositor
    surface_bounds_cache: Arc<Mutex<HashMap<SurfaceId, SurfaceBoundsEntry>>>,

    // Layout version counter (incremented when compositor runs)
    layout_version: Arc<AtomicU64>,

    // Per-layer persistent slab state (spec #94). The registry owns the
    // allocator and residency decisions; the buffers are the grow-only
    // storage the registry's ranges index into. Clean layers upload nothing;
    // dirty layers upload exactly their own slab.
    slab_registry: SlabRegistry,
    slab_buffers: SlabGpuBuffers,
    /// Frame-to-frame cache of slab bind groups; invalidated when the
    /// underlying buffers are recreated (see `ensure_slab_buffer_capacities`).
    slab_group_cache: SlabGroupCache,
    /// Reusable byte scratch for dirty-layer slab uploads.
    slab_upload_scratch: Vec<u8>,
    /// Reusable storage for per-frame dirty transform drains.
    transform_scratch: Vec<(u32, GpuLayerTransform)>,
    /// Persistent per-layer textures for texture-retained layers (#96),
    /// keyed by dense [`crate::LayerId`]. Created on first use, sampled by the
    /// surfaces pipeline on every clean composite frame, and dropped (with a
    /// re-record request) on resize or idleness.
    layer_textures: FxHashMap<crate::LayerId, LayerTextureEntry>,
    /// Monotonic frame counter for `layer_textures` idle eviction.
    layer_texture_frame: u64,
    // Last values pushed into the frame-constant uniform buffers, so an idle
    // window issues zero `write_buffer` calls at all.
    uploaded_globals: Option<GlobalParams>,
    uploaded_color_adjustments: Option<ColorAdjustments>,

    // Timestamp-query manager for flamegraph GPU capture (issue #57). Lazily
    // allocated only while a GPU-capturing session is active, so VRAM/setup
    // cost is zero otherwise. `blit_surfaces_direct` is a `&self` method, so
    // this needs shared-mutable access via `parking_lot::Mutex` rather than a
    // plain field.
    #[cfg(feature = "flamegraph")]
    gpu_query_manager: parking_lot::Mutex<Option<crate::flamegraph_gpu::GpuQueryManager>>,

    // On-demand GPU deep capture (issue #60). `None` except for the brief
    // window between a `flamegraph::request_deep_capture()` call firing on a
    // `draw()` and that capture's resource readback completing a few frames
    // later -- see `flamegraph_gpu`'s Phase 4 section doc comment for why
    // this is a completely separate, non-persistent path from
    // `gpu_query_manager` above rather than sharing its machinery.
    #[cfg(feature = "flamegraph")]
    deep_capture: parking_lot::Mutex<Option<crate::flamegraph_gpu::DeepCapturePendingReadback>>,
}

impl WgpuRenderer {
    pub fn new<WindowHandle>(
        context: Arc<WgpuContext>,
        window: WindowHandle,
        atlas: Arc<WgpuAtlas>,
        width: u32,
        height: u32,
        path_sample_count: u32,
    ) -> anyhow::Result<Self>
    where
        WindowHandle: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
    {
        let surface = unsafe {
            context
                .instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: Some(window.display_handle()?.as_raw()),
                    raw_window_handle: window.window_handle()?.as_raw(),
                })?
        };

        let surface_capabilities = surface.get_capabilities(&context.adapter);

        // NOTE(mdeand): The shaders (hsla_to_rgba) output sRGB values directly, so we need a
        // NOTE(mdeand): non-sRGB surface format to avoid a double linear-to-sRGB conversion.
        // NOTE(mdeand): Prefer a non-sRGB format; fall back to whatever is available.
        let format = surface_capabilities
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(surface_capabilities.formats[0]);

        let alpha_mode = if surface_capabilities
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
        {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else {
            surface_capabilities.alpha_modes[0]
        };

        // allow overriding vsync behaviour.  The default is `Fifo` (vsync
        // enabled) which is what `wgpu` considers the safest presentation mode.
        // Setting `GPUI_DISABLE_VSYNC=1` in the environment will switch to
        // `Immediate`, which drops frames at the display's full rate.  A more
        // fine‑grained control (`GPUI_PRESENT_MODE=mailbox|fifo|immediate`) is
        // also supported for experimentation.
        let present_mode = std::env::var("GPUI_PRESENT_MODE")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "mailbox" => Some(wgpu::PresentMode::Mailbox),
                "immediate" => Some(wgpu::PresentMode::Immediate),
                "fifo" => Some(wgpu::PresentMode::Fifo),
                _ => None,
            })
            .unwrap_or_else(|| {
                if std::env::var("GPUI_DISABLE_VSYNC").is_ok() {
                    wgpu::PresentMode::Immediate
                } else {
                    wgpu::PresentMode::Fifo
                }
            });

        #[cfg(feature = "flamegraph")]
        crate::set_present_mode(match present_mode {
            wgpu::PresentMode::Fifo => crate::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox => crate::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate => crate::PresentMode::Immediate,
            _ => crate::PresentMode::Other,
        });

        let surface_configuration = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            format,
            width,
            height,
            present_mode,
            alpha_mode,
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: context.desired_maximum_frame_latency,
        };

        let atlas_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let surface_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("surface_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let backdrop_blur_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("backdrop_blur_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let pipelines =
            WgpuPipelines::new(context.as_ref(), &surface_configuration, path_sample_count);

        // Create persistent framebuffer for browser-canvas-style blitting
        let persistent_framebuffer = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("persistent_framebuffer"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let persistent_framebuffer_view =
            persistent_framebuffer.create_view(&wgpu::TextureViewDescriptor::default());

        // Create backdrop blur texture for capturing framebuffer content
        let backdrop_blur_texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("backdrop_blur_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let backdrop_blur_texture_view =
            backdrop_blur_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let (group_textures, group_views) =
            create_filter_group_textures(&context.device, width, height, format);

        let slab_buffers = SlabGpuBuffers::new(
            &context.device,
            context.device.limits().min_uniform_buffer_offset_alignment,
        );

        Ok(Self {
            context: context.clone(),
            surface: ManuallyDrop::new(surface),
            surface_configuration,
            atlas,
            atlas_sampler,
            surface_sampler,
            backdrop_blur_sampler,
            pipelines,
            rendering_parameters: RenderingParameters::from_env(),
            surface_bind_groups: Mutex::new(HashMap::new()),
            persistent_framebuffer: Some(persistent_framebuffer),
            persistent_framebuffer_view: Some(persistent_framebuffer_view),
            backdrop_blur_texture: Some(backdrop_blur_texture),
            backdrop_blur_texture_view: Some(backdrop_blur_texture_view),
            group_textures,
            group_views,
            surface_bounds_cache: Arc::new(Mutex::new(HashMap::new())),
            layout_version: Arc::new(AtomicU64::new(0)),
            slab_registry: SlabRegistry::new(),
            slab_buffers,
            slab_group_cache: SlabGroupCache::default(),
            slab_upload_scratch: Vec::new(),
            transform_scratch: Vec::new(),
            layer_textures: FxHashMap::default(),
            layer_texture_frame: 0,
            uploaded_globals: None,
            uploaded_color_adjustments: None,
            #[cfg(feature = "flamegraph")]
            gpu_query_manager: parking_lot::Mutex::new(None),
            #[cfg(feature = "flamegraph")]
            deep_capture: parking_lot::Mutex::new(None),
        })
    }

    /// Reserve a timestamp-write pair against the current flamegraph GPU
    /// capture generation, if one is active and actively recording a frame.
    /// Returns `None` (cheaply, no wgpu resource allocation) otherwise, e.g.
    /// when no capture is running, or when `blit_surfaces_direct` runs
    /// outside a `draw()`-initiated frame.
    #[cfg(feature = "flamegraph")]
    fn reserve_gpu_timestamps(
        &self,
        name: crate::SpanName,
        pass_kind: crate::GpuPassKind,
    ) -> Option<crate::flamegraph_gpu::ReservedTimestamps> {
        self.gpu_query_manager.lock().as_mut()?.reserve_pair(name, pass_kind)
    }

    /// Reconfigure the swapchain, excluding external render threads for the
    /// duration.
    ///
    /// `Surface::configure` waits for the device to go idle and fails fatally
    /// with `GpuWaitTimeout` if anything submits during that wait. Surfaces
    /// handed to external render threads (`WgpuSurfaceHandle`) share this
    /// device and queue, so the exclusive guard here is what keeps a window
    /// resize from racing them -- see `WgpuContext::gpu_submit_lock`'s doc
    /// comment for the full mechanism.
    ///
    /// All `Surface::configure` calls in this renderer must go through here.
    fn reconfigure_surface(&self) {
        let _exclusive = self.context.gpu_submit_lock.write();
        self.surface
            .configure(&self.context.device, &self.surface_configuration);
    }

    // -------------------------------------------------------------------
    // Layer slabs (spec #94).
    // -------------------------------------------------------------------

    /// Grow slab storage to cover the allocator's arenas. Recreation loses
    /// bytes, so both events void residency: every layer re-uploads on its
    /// next sync rather than drawing against orphaned buffers.
    fn ensure_slab_buffer_capacities(&mut self) {
        let mut recreated_any_kind = false;
        for kind in SlabKind::ALL {
            let elements = self.slab_registry.arena_element_capacity(kind).max(MIN_CLASS);
            if self
                .slab_buffers
                .ensure_kind_capacity(&self.context.device, kind, elements)
            {
                recreated_any_kind = true;
                // The bind group still targets the orphaned buffer until it is
                // rebuilt against the new one.
                self.slab_group_cache.invalidate_kind(kind);
            }
        }
        if recreated_any_kind {
            log::info!("slab buffer grew; re-uploading all resident layers");
            self.slab_registry.invalidate_all_residency();
        }

        let slots_needed = self.slab_registry.transforms_shared().slot_count() + 1;
        if self
            .slab_buffers
            .ensure_transform_capacity(&self.context.device, slots_needed)
        {
            self.slab_registry.mark_all_transforms_dirty();
            self.slab_group_cache.invalidate_transforms();
        }
    }

    // -------------------------------------------------------------------
    // Layer textures (#96).
    // -------------------------------------------------------------------

    /// Make sure a persistent texture exists for `target` at the right size,
    /// creating (or recreating) it on first use and on resize. Returns the
    /// pixel size, or `None` for a degenerate extent.
    fn ensure_layer_texture(
        &mut self,
        target: &crate::scene::LayerTextureTarget,
    ) -> Option<(u32, u32)> {
        let width = target.texture_bounds.size.width.0.max(0.0).ceil() as u32;
        let height = target.texture_bounds.size.height.0.max(0.0).ceil() as u32;
        if width == 0 || height == 0 {
            return None;
        }
        if let Some(entry) = self.layer_textures.get(&target.layer_id) {
            if entry.width == width && entry.height == height {
                return Some((width, height));
            }
        }

        let format = self.surface_configuration.format;
        let texture = self.context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("layer_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        crate::render_stats::count("layer: texture allocated");
        log::trace!(
            "layer texture for {:?} (key {:?}) allocated at {width}x{height}",
            target.layer_id,
            target.key
        );
        // The view holds the texture alive; the texture handle itself is not
        // needed again until a size change recreates both.
        drop(texture);
        self.layer_textures.insert(
            target.layer_id,
            LayerTextureEntry {
                view,
                width,
                height,
                key: target.key,
                content_token: target.content_token,
                texture_bounds: target.texture_bounds,
                last_used_frame: self.layer_texture_frame,
            },
        );
        Some((width, height))
    }

    /// Drop layer textures that went unreferenced for
    /// [`LAYER_TEXTURE_IDLE_FRAMES`] frames, posting re-record requests so the
    /// layers re-bake on their next composite instead of sampling a missing
    /// texture.
    fn gc_layer_textures(&mut self) {
        let frame = self.layer_texture_frame;
        let stale: Vec<crate::LayerId> = self
            .layer_textures
            .iter()
            .filter(|(_, entry)| frame.saturating_sub(entry.last_used_frame) > LAYER_TEXTURE_IDLE_FRAMES)
            .map(|(id, _)| *id)
            .collect();
        for id in stale {
            if let Some(entry) = self.layer_textures.remove(&id) {
                log::trace!("layer texture for {id:?} (key {:?}) evicted idle", entry.key);
                self.slab_registry.request_rerecord([entry.key]);
            }
        }
    }

    /// One sync decision per layer per frame. Clean layers cost zero
    /// `write_buffer`s; a dirty layer uploads exactly its own slab ranges.
    /// Runs before the render pass so the pass below only draws.
    fn resolve_slab_spans(&mut self, scene: &Scene) {
        profiling::scope!("wgpui: slab sync");
        let pages_by_layer = collect_referenced_pages_by_layer(scene);
        let mut synced_layers: FxHashSet<LayerKey> = FxHashSet::default();
        for span in &scene.layer_slab_spans {
            if self.slab_registry.is_awaiting_rerecord(span.key) {
                continue;
            }
            if !synced_layers.insert(span.key) {
                continue;
            }
            match self
                .slab_registry
                .plan_sync(span.key, span.content_token, span.totals)
            {
                Err(error) => {
                    slab_gpu::report_sync_overflow(error);
                    self.slab_registry.request_rerecord([span.key]);
                    continue;
                }
                Ok(SyncPlan::Clean) => self.slab_registry.note_span_drawn_clean(),
                Ok(SyncPlan::UploadAllOccupied) => {
                    self.upload_layer_slab_bytes(scene, span.key);
                }
            }
            // A texture-retained layer's pack was built at the texture origin,
            // so its spans draw in texture space with an identity translate
            // (#96); everything else translates from layer-local to window.
            if let Some(target) = &span.texture {
                if self.ensure_layer_texture(target).is_none() {
                    self.slab_registry.request_rerecord([span.key]);
                    continue;
                }
                self.slab_registry.set_layer_translate(span.key, [0.0, 0.0]);
            } else {
                self.slab_registry.set_layer_translate(span.key, span.origin);
            }

            if let Some(pages) = pages_by_layer.get(&span.key) {
                self.slab_registry.note_referenced_pages(span.key, pages.iter().copied());
            }
        }

        let transforms_buffer = self.slab_buffers.transforms_buffer().clone();
        let stride = self.slab_buffers.transform_slot_stride;
        let queue = &self.context.queue;
        let mut dirty_transforms = std::mem::take(&mut self.transform_scratch);
        self.slab_registry.take_dirty_transforms_into(&mut dirty_transforms);
        for &(slot, transform) in &dirty_transforms {
            queue.write_buffer(
                &transforms_buffer,
                slot as u64 * stride,
                bytemuck::bytes_of(&transform),
            );
        }
        self.transform_scratch = dirty_transforms;
    }

    /// Upload every occupied kind range of one layer from its spans' packed
    /// arrays, at byte offsets derived from element-unit bases.
    ///
    /// The byte scratch is renderer-owned and reused across dirty syncs: a
    /// steady-state window re-uploads some layer every few frames, so a fresh
    /// `Vec` per sync would churn the allocator for bytes it just freed.
    fn upload_layer_slab_bytes(&mut self, scene: &Scene, key: LayerKey) {
        let Some(slabs) = self.slab_registry.entry_slabs(key) else {
            return;
        };
        let mut scratch = std::mem::take(&mut self.slab_upload_scratch);
        for kind in SlabKind::ALL {
            scratch.clear();
            for span in &scene.layer_slab_spans {
                if span.key != key {
                    continue;
                }
                append_packed_kind_bytes(&mut scratch, kind, &span.packed);
            }
            let range = slabs.slab(kind);
            if scratch.is_empty() || range.is_empty() {
                continue;
            }
            let stride = slab_gpu::instance_stride(kind);
            debug_assert_eq!(
                scratch.len() as u64,
                range.count as u64 * stride,
                "packed byte stream must match the reserved range"
            );
            self.context.queue.write_buffer(
                self.slab_buffers.kind_buffer(kind),
                range.byte_offset(stride),
                &scratch,
            );
            crate::render_stats::add(slab_gpu::COUNTER_BYTES_UPLOADED, scratch.len() as u64);
        }
        self.slab_upload_scratch = scratch;
    }

    /// Bind groups for one frame's slab draws: per-kind storage bindings over
    /// the slab buffers plus every distinct atlas texture any span references.
    ///
    /// Sourced from [`Self::slab_group_cache`], so a Clean-only frame clones
    /// handles instead of rebuilding bind groups; the cache is invalidated by
    /// [`Self::ensure_slab_buffer_capacities`] whenever a backing buffer is
    /// recreated, and page groups refresh only when the page set changes.
    fn slab_draw_groups_for_frame(&mut self, scene: &Scene) -> SlabDrawGroups {
        self.slab_group_cache.frame_groups(
            &self.context.device,
            &self.pipelines,
            &self.slab_buffers,
            &self.atlas,
            &self.atlas_sampler,
            scene,
        )
    }

    /// Draw one spliced layer's runs as instanced draws into the batch
    /// stream, merging adjacent same-kind same-state runs where free —
    /// including stretches that continue across consecutive span boundaries
    /// of the same layer via `open_run`.
    ///
    /// Any state that would produce wrong pixels — an entry awaiting a
    /// re-record after eviction, missing registry state — skips the draws
    /// loudly instead; the posted re-record request rebuilds the layer.
    #[allow(clippy::too_many_arguments)]
    fn draw_layer_slab_span(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        scene: &Scene,
        span_index: usize,
        groups: &SlabDrawGroups,
        transform_slot_stride: u64,
        state: &mut PassBindState,
        open_run: &mut Option<OpenSlabRun>,
    ) {
        let Some(span) = scene.layer_slab_spans.get(span_index) else {
            debug_assert!(false, "frame_batches yielded an out-of-range span index");
            return;
        };
        if self.slab_registry.is_awaiting_rerecord(span.key) {
            self.slab_registry.note_span_skipped_awaiting_rerecord();
            return;
        }
        let Some(slabs) = self.slab_registry.entry_slabs(span.key) else {
            slab_gpu::report_missing_slab_state(&self.slab_registry, span.key);
            return;
        };
        let Some(transform_slot) = self.slab_registry.transform_slot(span.key) else {
            slab_gpu::report_missing_slab_state(&self.slab_registry, span.key);
            return;
        };
        for run in &span.runs {
            if let Some(texture_id) = run.texture_id {
                let key = (texture_id.index, texture_id.kind);
                if !groups.sprite_textures.contains_key(&key) {
                    // The atlas page died between resolution and draw; treat
                    // it exactly like eviction poisoning.
                    self.slab_registry.request_rerecord([span.key]);
                    self.slab_registry.note_span_skipped_awaiting_rerecord();
                    return;
                }
            }
        }

        for run in &span.runs {
            // Continue an open stretch: same layer, same kind, same texture,
            // exactly contiguous in the layer-wide instance stream. Their
            // instances draw as one call because everything bound between
            // them is identical.
            if let Some(open) = open_run
                && open.accepts(span.key, &slabs, run)
            {
                open.count += run.count;
                continue;
            }
            flush_open_slab_run(
                &self.pipelines,
                transform_slot_stride,
                pass,
                groups,
                state,
                open_run,
            );
            *open_run = Some(OpenSlabRun {
                key: span.key,
                slabs,
                transform_slot,
                kind: run.kind,
                texture_id: run.texture_id,
                start: run.start,
                count: run.count,
            });
        }
    }

    /// Draw one texture-retained layer's span runs into the CURRENT pass —
    /// the layer-texture pass the caller set up (#96). No cross-run merging:
    /// a texture bake is a refill-frame path, and the pass is torn down right
    /// after.
    fn draw_texture_span_runs(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        span: &crate::scene::LayerSlabSpan,
        groups: &SlabDrawGroups,
        transform_slot_stride: u64,
        state: &mut PassBindState,
    ) {
        if self.slab_registry.is_awaiting_rerecord(span.key) {
            self.slab_registry.note_span_skipped_awaiting_rerecord();
            return;
        }
        let Some(slabs) = self.slab_registry.entry_slabs(span.key) else {
            slab_gpu::report_missing_slab_state(&self.slab_registry, span.key);
            return;
        };
        let Some(transform_slot) = self.slab_registry.transform_slot(span.key) else {
            slab_gpu::report_missing_slab_state(&self.slab_registry, span.key);
            return;
        };
        for run in &span.runs {
            if let Some(texture_id) = run.texture_id {
                let key = (texture_id.index, texture_id.kind);
                if !groups.sprite_textures.contains_key(&key) {
                    // The atlas page died between resolution and draw; treat
                    // it exactly like eviction poisoning.
                    self.slab_registry.request_rerecord([span.key]);
                    self.slab_registry.note_span_skipped_awaiting_rerecord();
                    return;
                }
            }
            let mut one = Some(OpenSlabRun {
                key: span.key,
                slabs,
                transform_slot,
                kind: run.kind,
                texture_id: run.texture_id,
                start: run.start,
                count: run.count,
            });
            flush_open_slab_run(
                &self.pipelines,
                transform_slot_stride,
                pass,
                groups,
                state,
                &mut one,
            );
        }
    }

    /// Drain this renderer's pending slab re-record requests.
    ///
    /// Called by the owning window at the start of its draw. Requests are
    /// per-renderer on purpose: a process-global queue would let another
    /// window's draw consume them, and a request that never reaches its owner
    /// leaves that owner's poisoned layers skipping draws indefinitely.
    pub fn take_rerecord_requests(&mut self) -> Vec<crate::LayerKey> {
        self.slab_registry.take_rerecord_requests()
    }

    pub fn draw(&mut self, scene: &Scene) {
        profiling::scope!("wgpui: renderer draw");
        log::trace!("Renderer::draw: starting frame");

        let mut command_encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("main"),
                });

        // Flamegraph GPU capture (issue #57): lazily create/tear down the
        // query manager to match whether a GPU-capturing session is active,
        // poll any in-flight readback from earlier frames (non-blocking, safe
        // on this thread — see flamegraph_gpu's module docs), then reserve
        // this frame's generation and bracket the whole encoder with a
        // GpuSubmitPresent span.
        #[cfg(feature = "flamegraph")]
        let flamegraph_submit_present = {
            {
                let mut guard = self.gpu_query_manager.lock();
                crate::flamegraph_gpu::sync_with_active_capture(
                    &mut guard,
                    &self.context.device,
                    &self.context.queue,
                );
                if let Some(manager) = guard.as_mut() {
                    manager.poll_readback(&self.context.device);
                    if let Some(frame_index) = crate::current_gpu_correlation_frame_index() {
                        manager.begin_frame(frame_index);
                    }
                }
            }

            let reserved = self.reserve_gpu_timestamps(
                crate::SpanName::Static("GpuSubmitPresent"),
                crate::GpuPassKind::SubmitPresent,
            );
            if let Some(reserved) = &reserved {
                command_encoder.write_timestamp(reserved.query_set(), reserved.begin_index());
            }
            reserved
        };

        // On-demand GPU deep capture (issue #60): harvest any previous deep
        // capture whose readback has completed (non-blocking, same
        // render-thread poll pattern as the query manager above), then arm a
        // new recorder for *this* frame if one was requested and none is
        // currently in flight. See `flamegraph_gpu`'s Phase 4 section doc
        // comment for the full lifecycle this participates in.
        #[cfg(feature = "flamegraph")]
        let mut deep_capture_recorder: Option<crate::flamegraph_gpu::DeepCaptureRecorder> = {
            let mut guard = self.deep_capture.lock();
            if let Some(pending) = guard.as_mut()
                && let Some(capture) = pending.poll(&self.context.device)
            {
                crate::flamegraph::complete_deep_capture(capture);
                *guard = None;
            }
            if guard.is_none() && crate::flamegraph::take_deep_capture_request() {
                Some(crate::flamegraph_gpu::DeepCaptureRecorder::new())
            } else {
                None
            }
        };

        self.atlas.before_frame(&mut command_encoder);
        log::trace!("Renderer::draw: atlas.before_frame complete");

        // Slab residency upkeep (spec #94): eviction poisoning first — stale
        // tile ids must never reach the GPU this frame — then arena growth,
        // then advisory compaction while the frame is otherwise idle.
        let evicted_pages = self.atlas.drain_destroyed_pages();
        if !evicted_pages.is_empty() {
            let poisoned = self.slab_registry.poison_on_evicted_pages(&evicted_pages);
            if !poisoned.is_empty() {
                self.slab_registry.request_rerecord(poisoned);
            }
        }
        self.slab_registry.begin_frame();
        self.ensure_slab_buffer_capacities();
        // Layer-texture upkeep (#96): age the frame counter, drop entries that
        // have gone unreferenced too long (posting re-record requests), so an
        // evicted buffer's texture does not hold VRAM forever.
        self.layer_texture_frame += 1;
        self.gc_layer_textures();
        // Advisory compaction, gated three ways (all scheduling, never
        // correctness): kill switch, utilization heuristic, and — since the
        // arenas never shrink and GC keeps utilization low regardless — a
        // zero-move backoff plus an uploads-in-flight deferral so an idle
        // window stops rebuilding empty plans every frame.
        if slab_gpu::compaction_enabled()
            && self.slab_registry.should_compact(0.35)
            && self.slab_registry.compaction_gate_open()
        {
            let plan = self.slab_registry.compaction_plan();
            let moves = self.slab_registry.apply_compaction(&plan);
            if moves.is_empty() {
                self.slab_registry.note_zero_move_plan();
            } else {
                self.slab_registry.note_moves_applied();
            }
            for (kind, src, dst) in moves {
                let stride = slab_gpu::instance_stride(kind);
                command_encoder.copy_buffer_to_buffer(
                    self.slab_buffers.kind_buffer(kind),
                    src.byte_offset(stride),
                    self.slab_buffers.kind_buffer(kind),
                    dst.byte_offset(stride),
                    src.count as u64 * stride,
                );
            }
        }


        // keep track of which surface ids we rendered this frame
        let mut seen_surfaces: Vec<crate::platform::cross::surface_registry::SurfaceId> =
            Vec::new();

        // CRITICAL: Keep surface views alive until after the render pass ends
        // The bind groups reference these views, so they must not be dropped early
        let mut surface_views: Vec<wgpu::TextureView> = Vec::new();
        // Surface bind groups also reference per-surface params buffers.
        let mut surface_param_buffers: Vec<wgpu::Buffer> = Vec::new();

        // Covers every per-frame `write_buffer` below, up to the point the
        // swapchain image is acquired — with slabs live this is the residual
        // cost for legacy (unspliced) content only; clean slabbed layers
        // contribute nothing.
        {
            profiling::scope!("wgpui: gpu upload");
            let gpu_upload_timer = crate::render_stats::scope("frame: gpu upload");

            let color_adjustments = ColorAdjustments {
                gamma_ratios: self.rendering_parameters.gamma_ratios,
                grayscale_enhanced_contrast: self.rendering_parameters.grayscale_enhanced_contrast,
                _padding: [0.0; 3],
            };
            if self.uploaded_color_adjustments != Some(color_adjustments) {
                self.context.queue.write_buffer(
                    &self.context.color_adjustments_buffer,
                    0,
                    bytemuck::bytes_of(&color_adjustments),
                );
                self.uploaded_color_adjustments = Some(color_adjustments);
            }

            let globals = GlobalParams {
                viewport_size: [
                    self.surface_configuration.width as f32,
                    self.surface_configuration.height as f32,
                ],
                premultimated_alpha: match self.surface_configuration.alpha_mode {
                    wgpu::CompositeAlphaMode::PreMultiplied => 1,
                    _ => 0,
                },
                pad: 0,
            };

            if self.uploaded_globals != Some(globals) {
                self.context.queue.write_buffer(
                    &self.context.globals_buffer,
                    0,
                    bytemuck::bytes_of(&globals),
                );
                self.uploaded_globals = Some(globals);
            }

            if !scene.quads.is_empty() {
                let data = bytemuck::cast_slice(&scene.quads);
                ensure_buffer_size(
                    &self.context.device,
                    &self.context.quads_buffer,
                    data.len() as u64,
                    "Quads Buffer",
                    wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::STORAGE,
                );
                self.context
                    .queue
                    .write_buffer(&self.context.quads_buffer.lock(), 0, data);
            }
            if !scene.shadows.is_empty() {
                let data = bytemuck::cast_slice(&scene.shadows);
                ensure_buffer_size(
                    &self.context.device,
                    &self.context.shadows_buffer,
                    data.len() as u64,
                    "Shadows Buffer",
                    wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::STORAGE,
                );
                self.context
                    .queue
                    .write_buffer(&self.context.shadows_buffer.lock(), 0, data);
            }
            if !scene.backdrop_filters.is_empty() {
                let data = bytemuck::cast_slice(&scene.backdrop_filters);
                ensure_buffer_size(
                    &self.context.device,
                    &self.context.backdrop_filters_buffer,
                    data.len() as u64,
                    "Backdrop Filters Buffer",
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                );
                self.context.queue.write_buffer(
                    &self.context.backdrop_filters_buffer.lock(),
                    0,
                    data,
                );
            }
            if !scene.underlines.is_empty() {
                let data = bytemuck::cast_slice(&scene.underlines);
                ensure_buffer_size(
                    &self.context.device,
                    &self.context.underlines_buffer,
                    data.len() as u64,
                    "Underlines Buffer",
                    wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::STORAGE,
                );
                self.context.queue.write_buffer(
                    &self.context.underlines_buffer.lock(),
                    0,
                    data,
                );
            }
            if !scene.monochrome_sprites.is_empty() {
                let data = bytemuck::cast_slice(&scene.monochrome_sprites);
                ensure_buffer_size(
                    &self.context.device,
                    &self.context.mono_sprites_buffer,
                    data.len() as u64,
                    "Monosprites Buffer",
                    wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::STORAGE,
                );
                self.context.queue.write_buffer(
                    &self.context.mono_sprites_buffer.lock(),
                    0,
                    data,
                );
            }
            if !scene.polychrome_sprites.is_empty() {
                let data = bytemuck::cast_slice(&scene.polychrome_sprites);
                ensure_buffer_size(
                    &self.context.device,
                    &self.context.poly_sprites_buffer,
                    data.len() as u64,
                    "Poly Sprites Buffer",
                    wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::STORAGE,
                );
                self.context.queue.write_buffer(
                    &self.context.poly_sprites_buffer.lock(),
                    0,
                    data,
                );
            }

            // Build flat vertex array for all paths (color + content mask baked per-vertex)
            let mut flat_path_vertices: Vec<GpuPathVertex> = Vec::new();
            for path in &scene.paths {
                let color = path.color.solid;
                let cm = &path.content_mask.bounds;
                let cm_origin = [cm.origin.x.0, cm.origin.y.0];
                let cm_size = [cm.size.width.0, cm.size.height.0];
                for vertex in &path.vertices {
                    flat_path_vertices.push(GpuPathVertex {
                        xy_position: [vertex.xy_position.x.0, vertex.xy_position.y.0],
                        st_position: [vertex.st_position.x, vertex.st_position.y],
                        hsla: [color.h, color.s, color.l, color.a],
                        content_mask_origin: cm_origin,
                        content_mask_size: cm_size,
                    });
                }
            }
            if !flat_path_vertices.is_empty() {
                let data = bytemuck::cast_slice(&flat_path_vertices);
                ensure_buffer_size(
                    &self.context.device,
                    &self.context.paths_vertices_buffer,
                    data.len() as u64,
                    "Path Vertices Buffer",
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                );
                self.context.queue.write_buffer(
                    &self.context.paths_vertices_buffer.lock(),
                    0,
                    data,
                );
            }

            // Slab span resolution happens inside the upload timer's scope: it is
            // exactly the per-frame upload work, and clean layers resolve here
            // with zero queue writes.
            self.resolve_slab_spans(scene);

            drop(gpu_upload_timer);
        }

        // Acquire the next swapchain image.  On the first frame after window
        // creation (or after a resize races with the GPU) the surface can be
        // reported as `Outdated` or `Other`.  Rather than panicking we
        // reconfigure and retry once; if the second attempt also fails we
        // simply drop this frame.
        let surface_texture = {
            match self.surface.get_current_texture() {
                CurrentSurfaceTexture::Success(t)
                | CurrentSurfaceTexture::Suboptimal(t) => t,
                CurrentSurfaceTexture::Outdated
                | CurrentSurfaceTexture::Lost
                | CurrentSurfaceTexture::Validation => {
                    // Reconfigure with the current known size and retry.
                    self.reconfigure_surface();
                    match self.surface.get_current_texture() {
                        CurrentSurfaceTexture::Success(t)
                        | CurrentSurfaceTexture::Suboptimal(t) => t,
                        other => {
                            log::warn!(
                                "Skipping frame: failed to acquire swap chain texture after reconfigure: {:?}",
                                other
                            );
                            return;
                        }
                    }
                }
                CurrentSurfaceTexture::Timeout => {
                    log::warn!("Skipping frame: swap chain acquire timed out");
                    return;
                }
                CurrentSurfaceTexture::Occluded => {
                    log::warn!("Skipping frame: swap chain acquire occluded");
                    return;
                }
            }
        };

        // Increment layout version - all bounds caches are now fresh
        // IMPORTANT: Only increment after successful swapchain acquisition
        // If we skip the frame, bounds remain valid
        self.layout_version.fetch_add(1, Ordering::Release);

        // Slab bind state comes from the frame-to-frame cache, which needs
        // `&mut` access to the cache field — built before the legacy buffer
        // locks below are held for the rest of the function.
        let layer_transform_bind_group = self
            .slab_group_cache
            .transforms_group(&self.context.device, &self.pipelines, &self.slab_buffers);

        // Only frames that actually carry spans pay for slab bind state; the
        // cache makes even those frames cheap when nothing was recreated.
        let slab_groups =
            (scene.slab_span_count() > 0).then(|| self.slab_draw_groups_for_frame(scene));

        // Borrow buffers for bind group creation - these borrows must live until bind groups are done
        let quads_buffer_ref = self.context.quads_buffer.lock();
        let shadows_buffer_ref = self.context.shadows_buffer.lock();
        let backdrop_filters_buffer_ref = self.context.backdrop_filters_buffer.lock();
        let underlines_buffer_ref = self.context.underlines_buffer.lock();
        let mono_sprites_buffer_ref = self.context.mono_sprites_buffer.lock();
        let poly_sprites_buffer_ref = self.context.poly_sprites_buffer.lock();
        let paths_vertices_buffer_ref = self.context.paths_vertices_buffer.lock();

        let quads_bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("quads_bind_group"),
                layout: &self.pipelines.quads_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &quads_buffer_ref,
                        offset: 0,
                        size: None,
                    }),
                }],
            });

        let shadows_bind_group =
            self.context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadows_bind_group"),
                    layout: &self.pipelines.shadows_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &shadows_buffer_ref,
                            offset: 0,
                            size: None,
                        }),
                    }],
                });

        let backdrop_filters_bind_group =
            self.context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("backdrop_filters_bind_group"),
                    layout: &self.pipelines.backdrop_filters_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &backdrop_filters_buffer_ref,
                            offset: 0,
                            size: None,
                        }),
                    }],
                });

        let backdrop_texture_bind_group =
            self.context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("backdrop_texture_bind_group"),
                    layout: &self.pipelines.backdrop_texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                self.backdrop_blur_texture_view.as_ref().unwrap(),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.backdrop_blur_sampler),
                        },
                    ],
                });

        let underlines_bind_group =
            self.context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("underlines_bind_group"),
                    layout: &self.pipelines.underlines_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &underlines_buffer_ref,
                            offset: 0,
                            size: None,
                        }),
                    }],
                });

        let mono_sprites_bind_group =
            self.context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mono_sprites_bind_group"),
                    layout: &self.pipelines.mono_sprites_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &mono_sprites_buffer_ref,
                            offset: 0,
                            size: None,
                        }),
                    }],
                });

        let poly_sprites_bind_group =
            self.context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("poly_sprites_bind_group"),
                    layout: &self.pipelines.poly_sprites_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &poly_sprites_buffer_ref,
                            offset: 0,
                            size: None,
                        }),
                    }],
                });

        let paths_bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("paths_bind_group"),
                layout: &self.pipelines.paths_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &paths_vertices_buffer_ref,
                        offset: 0,
                        size: None,
                    }),
                }],
            });

        {
            #[cfg(feature = "flamegraph")]
            let flamegraph_main_pass =
                self.reserve_gpu_timestamps(crate::SpanName::Static("main"), crate::GpuPassKind::Main);

            let mut pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self
                        .persistent_framebuffer_view
                        .as_ref()
                        .expect("persistent framebuffer view must exist"),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    resolve_target: None,
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                #[cfg(feature = "flamegraph")]
                timestamp_writes: flamegraph_main_pass.as_ref().map(|reserved| reserved.writes()),
                #[cfg(not(feature = "flamegraph"))]
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let mut quads_first_instance: u32 = 0;
            let mut shadows_first_instance: u32 = 0;
            let mut backdrop_filters_first_instance: u32 = 0;
            let mut underlines_first_instance: u32 = 0;
            let mut mono_sprites_first_instance: u32 = 0;
            let mut poly_sprites_first_instance: u32 = 0;
            let mut paths_vertex_offset: u32 = 0;

            // Stack of active content-filter groups. Each entry pairs the group's
            // `FilterBoundary` start marker with the `group_textures`/`group_views`
            // slot its content is being rendered into (`None` if the group exceeded
            // `MAX_FILTER_DEPTH` and is being painted inline, unisolated).
            let mut filter_stack: Vec<(FilterBoundary, Option<usize>)> = Vec::new();

            // Deep capture (issue #60): tracks which render pass `pass`
            // currently points at, so each recorded draw call's `pass_label`
            // reflects reality even though `pass` gets reassigned mid-loop
            // (backdrop filters/filter groups end and re-begin it). Kept
            // outside the `#[cfg(feature = "flamegraph")]` recorder itself
            // since it's just a `&'static str`, cheaper than gating every one
            // of its several assignment sites.
            #[cfg(feature = "flamegraph")]
            let mut current_pass_label: &'static str = "main";

            // Bind-state tracker and cross-span merge slot for the whole
            // main pass. Both must reset wherever the pass is dropped and
            // re-begun (backdrop filters, filter groups) or where draws use
            // bind groups this module does not model (surfaces, filter
            // composites).
            let transform_slot_stride = self.slab_buffers.transform_slot_stride;
            let mut pass_state = PassBindState::default();
            let mut open_slab_run: Option<OpenSlabRun> = None;

            // Layer textures (#96) already cleared this frame: the first span
            // of a layer clears its texture, later spans load and accumulate.
            let mut layer_textures_cleared: FxHashSet<crate::LayerId> = FxHashSet::default();

            for frame_batch in scene.frame_batches() {
                let batch = match frame_batch {
                    crate::scene::SceneBatch::Primitives(batch) => {
                        // A pending slab stretch precedes this draw in paint
                        // order; flush it into the same pass before anything
                        // else renders. Zero-instance batches (empty split
                        // halves) draw nothing and must not break adjacency.
                        if primitive_batch_instance_count(&batch) > 0
                            && open_slab_run.is_some()
                            && let Some(groups) = slab_groups.as_ref()
                        {
                            flush_open_slab_run(
                                &self.pipelines,
                                transform_slot_stride,
                                &mut pass,
                                groups,
                                &mut pass_state,
                                &mut open_slab_run,
                            );
                        }
                        batch
                    }
                    crate::scene::SceneBatch::LayerSlab(span_index) => {
                        let texture_target = scene
                            .layer_slab_spans
                            .get(span_index)
                            .and_then(|span| span.texture.clone());
                        if let Some(target) = texture_target {
                            // #96: this span bakes a texture-retained layer's
                            // content into its persistent texture. Flush the
                            // main pass's open run, redirect into the layer
                            // texture, then resume the main pass untouched.
                            if let Some(groups) = slab_groups.as_ref() {
                                flush_open_slab_run(
                                    &self.pipelines,
                                    transform_slot_stride,
                                    &mut pass,
                                    groups,
                                    &mut pass_state,
                                    &mut open_slab_run,
                                );
                            }
                            drop(pass);

                            // `resolve_slab_spans` already created the texture
                            // (and posted a re-record if it could not); this
                            // loop holds immutable buffer locks, so it only
                            // reads the cache here.
                            let texture_ready = {
                                let width = target.texture_bounds.size.width.0.max(0.0).ceil() as u32;
                                let height =
                                    target.texture_bounds.size.height.0.max(0.0).ceil() as u32;
                                self.layer_textures.get(&target.layer_id).is_some_and(|entry| {
                                    entry.width == width && entry.height == height
                                })
                            };
                            if !texture_ready {
                                pass = command_encoder.begin_render_pass(
                                    &wgpu::RenderPassDescriptor {
                                        label: Some("main"),
                                        color_attachments: &[Some(
                                            wgpu::RenderPassColorAttachment {
                                                view: self
                                                    .persistent_framebuffer_view
                                                    .as_ref()
                                                    .expect("framebuffer exists during draw"),
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Load,
                                                    store: wgpu::StoreOp::Store,
                                                },
                                                resolve_target: None,
                                                depth_slice: None,
                                            },
                                        )],
                                        depth_stencil_attachment: None,
                                        #[cfg(feature = "flamegraph")]
                                        timestamp_writes: None,
                                        #[cfg(not(feature = "flamegraph"))]
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                        multiview_mask: None,
                                    },
                                );
                                #[cfg(feature = "flamegraph")]
                                {
                                    current_pass_label = "main";
                                }
                                pass_state.reset();
                                continue;
                            }
                            let texture_view = self.layer_textures[&target.layer_id].view.clone();
                            if let Some(entry) = self.layer_textures.get_mut(&target.layer_id) {
                                entry.last_used_frame = self.layer_texture_frame;
                                entry.content_token = target.content_token;
                            }
                            let clear = layer_textures_cleared.insert(target.layer_id);

                            #[cfg(feature = "flamegraph")]
                            let flamegraph_layer_texture_pass = self.reserve_gpu_timestamps(
                                crate::SpanName::Static("layer_texture"),
                                crate::GpuPassKind::FilterGroup,
                            );
                            pass = command_encoder.begin_render_pass(
                                &wgpu::RenderPassDescriptor {
                                    label: Some("layer_texture"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &texture_view,
                                        ops: wgpu::Operations {
                                            load: if clear {
                                                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                                            } else {
                                                wgpu::LoadOp::Load
                                            },
                                            store: wgpu::StoreOp::Store,
                                        },
                                        resolve_target: None,
                                        depth_slice: None,
                                    })],
                                    depth_stencil_attachment: None,
                                    #[cfg(feature = "flamegraph")]
                                    timestamp_writes: flamegraph_layer_texture_pass
                                        .as_ref()
                                        .map(|reserved| reserved.writes()),
                                    #[cfg(not(feature = "flamegraph"))]
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                    multiview_mask: None,
                                },
                            );
                            #[cfg(feature = "flamegraph")]
                            {
                                current_pass_label = "layer_texture";
                            }
                            pass_state.reset();

                            // The packed coordinates are texture-relative, but
                            // every slab vertex shader maps window pixels
                            // through NDC using the window size. A viewport
                            // whose affine maps window space onto the texture
                            // (origin at -texture_origin, size = window) makes
                            // the existing pipelines draw texture-space
                            // geometry exactly where the texture lives — no
                            // shader variants needed.
                            let viewport_size = self.surface_configuration.width as f32;
                            let viewport_height = self.surface_configuration.height as f32;
                            pass.set_viewport(
                                -target.texture_bounds.origin.x.0,
                                -target.texture_bounds.origin.y.0,
                                viewport_size,
                                viewport_height,
                                0.0,
                                1.0,
                            );
                            pass.set_scissor_rect(0, 0, target.texture_bounds.size.width.0.ceil() as u32, target.texture_bounds.size.height.0.ceil() as u32);

                            if let Some(groups) = slab_groups.as_ref() {
                                if let Some(span) = scene.layer_slab_spans.get(span_index) {
                                    self.draw_texture_span_runs(
                                        &mut pass,
                                        span,
                                        groups,
                                        transform_slot_stride,
                                        &mut pass_state,
                                    );
                                }
                            }

                            // Resume the main pass where the redirect left it.
                            drop(pass);
                            pass = command_encoder.begin_render_pass(
                                &wgpu::RenderPassDescriptor {
                                    label: Some("main"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: self
                                            .persistent_framebuffer_view
                                            .as_ref()
                                            .expect("framebuffer exists during draw"),
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Load,
                                            store: wgpu::StoreOp::Store,
                                        },
                                        resolve_target: None,
                                        depth_slice: None,
                                    })],
                                    depth_stencil_attachment: None,
                                    #[cfg(feature = "flamegraph")]
                                    timestamp_writes: None,
                                    #[cfg(not(feature = "flamegraph"))]
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                    multiview_mask: None,
                                },
                            );
                            #[cfg(feature = "flamegraph")]
                            {
                                current_pass_label = "main";
                            }
                            pass_state.reset();
                            continue;
                        }
                        if let Some(groups) = slab_groups.as_ref() {
                            self.draw_layer_slab_span(
                                &mut pass,
                                scene,
                                span_index,
                                groups,
                                transform_slot_stride,
                                &mut pass_state,
                                &mut open_slab_run,
                            );
                        }
                        continue;
                    }
                };
                match batch {
                    PrimitiveBatch::Quads(quads) => {
                        let count = quads.len() as u32;
                        pass_state.set_pipeline(&mut pass, DrawPipelineId::Quads, &self.pipelines.quads_pipeline);
                        pass_state.set_bind_group(&mut pass, 0, BoundGroupId::Globals, &self.pipelines.globals_bind_group, &[]);
                        pass_state.set_bind_group(&mut pass, 1, BoundGroupId::LegacyBuffer(LegacyBuffer::Quads), &quads_bind_group, &[]);
                        // Dynamic offset 0: the permanently-zero identity
                        // slot, so absolute legacy coordinates draw unshifted.
                        pass_state.set_bind_group(&mut pass, 2, BoundGroupId::LayerTransform(0), &layer_transform_bind_group, &[0]);
                        pass.draw(0..4, quads_first_instance..quads_first_instance + count);
                        quads_first_instance += count;
                        #[cfg(feature = "flamegraph")]
                        crate::record_draw_call(crate::DrawCallKind::Quads, count);
                        #[cfg(feature = "flamegraph")]
                        if let Some(recorder) = deep_capture_recorder.as_mut() {
                            recorder.record_draw_call(
                                crate::DrawCallKind::Quads,
                                "quads",
                                current_pass_label,
                                0..4,
                                quads_first_instance - count..quads_first_instance,
                                2,
                                Some(crate::flamegraph::DeepCaptureBufferKind::Quads),
                                None,
                                None,
                            );
                        }
                    }

                    PrimitiveBatch::MonochromeSprites {
                        texture_id,
                        sprites,
                    } => {
                        let count = sprites.len() as u32;
                        let tex_info = self.atlas.get_texture_info(texture_id);

                        let sprites_texture_bind_group =
                            self.context
                                .device
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("sprites_bind_group"),
                                    layout: &self.pipelines.sprites_bind_group_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: wgpu::BindingResource::TextureView(
                                                &tex_info.raw_view,
                                            ),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::BindingResource::Sampler(
                                                &self.atlas_sampler,
                                            ),
                                        },
                                    ],
                                });

                        pass_state.set_pipeline(&mut pass, DrawPipelineId::MonoSprites, &self.pipelines.mono_sprites_pipeline);
                        pass_state.set_bind_group(&mut pass, 0, BoundGroupId::Globals, &self.pipelines.globals_bind_group, &[]);
                        pass_state.set_bind_group(&mut pass, 1, BoundGroupId::ColorAdjustments, &self.pipelines.color_adjustments_bind_group, &[]);
                        pass_state.set_bind_group(
                            &mut pass,
                            2,
                            BoundGroupId::SpriteTexture(texture_id.index, texture_id.kind),
                            &sprites_texture_bind_group,
                            &[],
                        );
                        pass_state.set_bind_group(&mut pass, 3, BoundGroupId::LegacyBuffer(LegacyBuffer::MonoSprites), &mono_sprites_bind_group, &[]);
                        pass_state.set_bind_group(&mut pass, 4, BoundGroupId::LayerTransform(0), &layer_transform_bind_group, &[0]);
                        pass.draw(
                            0..4,
                            mono_sprites_first_instance..mono_sprites_first_instance + count,
                        );
                        mono_sprites_first_instance += count;
                        #[cfg(feature = "flamegraph")]
                        crate::record_draw_call(crate::DrawCallKind::MonoSprites, count);
                        #[cfg(feature = "flamegraph")]
                        if let Some(recorder) = deep_capture_recorder.as_mut() {
                            recorder.record_draw_call(
                                crate::DrawCallKind::MonoSprites,
                                "mono_sprites",
                                current_pass_label,
                                0..4,
                                mono_sprites_first_instance - count..mono_sprites_first_instance,
                                4,
                                Some(crate::flamegraph::DeepCaptureBufferKind::MonoSprites),
                                Some(((texture_id.kind as u64) << 32) | texture_id.index as u64),
                                None,
                            );
                        }
                    }
                    PrimitiveBatch::PolychromeSprites {
                        texture_id,
                        sprites,
                    } => {
                        let count = sprites.len() as u32;
                        let tex_info = self.atlas.get_texture_info(texture_id);

                        let sprites_texture_bind_group =
                            self.context
                                .device
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("poly_sprites_texture_bind_group"),
                                    layout: &self.pipelines.sprites_bind_group_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: wgpu::BindingResource::TextureView(
                                                &tex_info.raw_view,
                                            ),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::BindingResource::Sampler(
                                                &self.atlas_sampler,
                                            ),
                                        },
                                    ],
                                });

                        pass_state.set_pipeline(&mut pass, DrawPipelineId::PolySprites, &self.pipelines.poly_sprites_pipeline);
                        pass_state.set_bind_group(&mut pass, 0, BoundGroupId::Globals, &self.pipelines.globals_bind_group, &[]);
                        pass_state.set_bind_group(
                            &mut pass,
                            1,
                            BoundGroupId::SpriteTexture(texture_id.index, texture_id.kind),
                            &sprites_texture_bind_group,
                            &[],
                        );
                        pass_state.set_bind_group(&mut pass, 2, BoundGroupId::LegacyBuffer(LegacyBuffer::PolySprites), &poly_sprites_bind_group, &[]);
                        pass_state.set_bind_group(&mut pass, 3, BoundGroupId::LayerTransform(0), &layer_transform_bind_group, &[0]);
                        pass.draw(
                            0..4,
                            poly_sprites_first_instance..poly_sprites_first_instance + count,
                        );
                        poly_sprites_first_instance += count;
                        #[cfg(feature = "flamegraph")]
                        crate::record_draw_call(crate::DrawCallKind::PolySprites, count);
                        #[cfg(feature = "flamegraph")]
                        if let Some(recorder) = deep_capture_recorder.as_mut() {
                            recorder.record_draw_call(
                                crate::DrawCallKind::PolySprites,
                                "poly_sprites",
                                current_pass_label,
                                0..4,
                                poly_sprites_first_instance - count..poly_sprites_first_instance,
                                3,
                                Some(crate::flamegraph::DeepCaptureBufferKind::PolySprites),
                                Some(((texture_id.kind as u64) << 32) | texture_id.index as u64),
                                None,
                            );
                        }
                    }
                    PrimitiveBatch::Shadows(shadows) => {
                        let count = shadows.len() as u32;
                        pass_state.set_pipeline(&mut pass, DrawPipelineId::Shadows, &self.pipelines.shadows_pipeline);
                        pass_state.set_bind_group(&mut pass, 0, BoundGroupId::Globals, &self.pipelines.globals_bind_group, &[]);
                        pass_state.set_bind_group(&mut pass, 1, BoundGroupId::LegacyBuffer(LegacyBuffer::Shadows), &shadows_bind_group, &[]);
                        pass_state.set_bind_group(&mut pass, 2, BoundGroupId::LayerTransform(0), &layer_transform_bind_group, &[0]);
                        pass.draw(0..4, shadows_first_instance..shadows_first_instance + count);
                        shadows_first_instance += count;
                        #[cfg(feature = "flamegraph")]
                        crate::record_draw_call(crate::DrawCallKind::Shadows, count);
                        #[cfg(feature = "flamegraph")]
                        if let Some(recorder) = deep_capture_recorder.as_mut() {
                            recorder.record_draw_call(
                                crate::DrawCallKind::Shadows,
                                "shadows",
                                current_pass_label,
                                0..4,
                                shadows_first_instance - count..shadows_first_instance,
                                2,
                                Some(crate::flamegraph::DeepCaptureBufferKind::Shadows),
                                None,
                                None,
                            );
                        }
                    }
                    PrimitiveBatch::BackdropFilters(backdrop_filters) => {
                        let count = backdrop_filters.len() as u32;

                        // End the current render pass to copy texture
                        drop(pass);

                        // Copy surface texture to backdrop_blur_texture for sampling
                        if let Some(ref blur_texture) = self.backdrop_blur_texture {
                            // Use actual surface texture size (may differ from configured size)
                            let surface_size = surface_texture.texture.size();

                            // Only copy if sizes match (otherwise skip to avoid validation error)
                            if surface_size.width == blur_texture.width()
                                && surface_size.height == blur_texture.height()
                            {
                                command_encoder.copy_texture_to_texture(
                                    surface_texture.texture.as_image_copy(),
                                    blur_texture.as_image_copy(),
                                    surface_size,
                                );
                            }
                        }

                        // Begin new render pass with Load to preserve existing content
                        #[cfg(feature = "flamegraph")]
                        let flamegraph_main_resumed_pass = self.reserve_gpu_timestamps(
                            crate::SpanName::Static("main_resumed"),
                            crate::GpuPassKind::MainResumed,
                        );
                        pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("main_resumed"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &surface_texture
                                    .texture
                                    .create_view(&wgpu::TextureViewDescriptor::default()),
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                                resolve_target: None,
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            #[cfg(feature = "flamegraph")]
                            timestamp_writes: flamegraph_main_resumed_pass.as_ref().map(|reserved| reserved.writes()),
                            #[cfg(not(feature = "flamegraph"))]
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        #[cfg(feature = "flamegraph")]
                        {
                            current_pass_label = "main_resumed";
                        }
                        // Fresh pass: nothing is bound anymore.
                        pass_state.reset();

                        // Now render the backdrop blur quads
                        pass.set_pipeline(&self.pipelines.backdrop_filters_pipeline);
                        pass.set_bind_group(0, &self.pipelines.globals_bind_group, &[]);
                        pass.set_bind_group(1, &backdrop_filters_bind_group, &[]);
                        pass.set_bind_group(2, &backdrop_texture_bind_group, &[]);
                        pass.draw(
                            0..4,
                            backdrop_filters_first_instance..backdrop_filters_first_instance + count,
                        );
                        backdrop_filters_first_instance += count;
                        #[cfg(feature = "flamegraph")]
                        crate::record_draw_call(crate::DrawCallKind::BackdropFilters, count);
                        #[cfg(feature = "flamegraph")]
                        if let Some(recorder) = deep_capture_recorder.as_mut() {
                            recorder.record_draw_call(
                                crate::DrawCallKind::BackdropFilters,
                                "backdrop_filters",
                                current_pass_label,
                                0..4,
                                backdrop_filters_first_instance - count..backdrop_filters_first_instance,
                                3,
                                Some(crate::flamegraph::DeepCaptureBufferKind::BackdropFilters),
                                None,
                                None,
                            );
                        }
                    }
                    PrimitiveBatch::FilterBoundary(index) => {
                        let boundary = scene.filter_boundaries[index];

                        if boundary.is_start {
                            let depth = filter_stack.len();
                            if depth >= self.group_textures.len() {
                                // Exceeded the supported nesting depth: paint the group's
                                // content inline (unisolated/unblurred) rather than dropping it.
                                filter_stack.push((boundary, None));
                            } else {
                                drop(pass);

                                #[cfg(feature = "flamegraph")]
                                let flamegraph_filter_group_pass = self.reserve_gpu_timestamps(
                                    crate::SpanName::Static("filter_group"),
                                    crate::GpuPassKind::FilterGroup,
                                );
                                pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("filter_group"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &self.group_views[depth],
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                            store: wgpu::StoreOp::Store,
                                        },
                                        resolve_target: None,
                                        depth_slice: None,
                                    })],
                                    depth_stencil_attachment: None,
                                    #[cfg(feature = "flamegraph")]
                                    timestamp_writes: flamegraph_filter_group_pass.as_ref().map(|reserved| reserved.writes()),
                                    #[cfg(not(feature = "flamegraph"))]
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                    multiview_mask: None,
                                });
                                #[cfg(feature = "flamegraph")]
                                {
                                    current_pass_label = "filter_group";
                                }
                                // Fresh pass: nothing is bound anymore.
                                pass_state.reset();

                                filter_stack.push((boundary, Some(depth)));
                            }
                        } else {
                            let Some((start_boundary, depth)) = filter_stack.pop() else {
                                continue;
                            };

                            let Some(depth) = depth else {
                                // The group was painted inline; nothing to composite.
                                continue;
                            };

                            // End the group's pass: its content is now baked into
                            // `group_textures[depth]`.
                            drop(pass);

                            let surface_view = surface_texture
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            let parent_view: &wgpu::TextureView = match filter_stack.last() {
                                Some((_, Some(parent_depth))) => &self.group_views[*parent_depth],
                                _ => &surface_view,
                            };

                            #[cfg(feature = "flamegraph")]
                            let flamegraph_filter_group_resumed_pass = self.reserve_gpu_timestamps(
                                crate::SpanName::Static("filter_group_resumed"),
                                crate::GpuPassKind::FilterGroupResumed,
                            );
                            pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("filter_group_resumed"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: parent_view,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                    resolve_target: None,
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                #[cfg(feature = "flamegraph")]
                                timestamp_writes: flamegraph_filter_group_resumed_pass.as_ref().map(|reserved| reserved.writes()),
                                #[cfg(not(feature = "flamegraph"))]
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });
                            #[cfg(feature = "flamegraph")]
                            {
                                current_pass_label = "filter_group_resumed";
                            }
                            // Fresh pass: nothing is bound anymore.
                            pass_state.reset();

                            // Composite the blurred group content back over the parent using
                            // the same backdrop-filter pipeline, sampling from the group's
                            // offscreen texture instead of a surface snapshot.
                            let composite = BackdropFilter {
                                order: 0,
                                bounds: start_boundary.bounds,
                                content_mask: start_boundary.content_mask,
                                corner_radii: start_boundary.corner_radii,
                                blur_radius: start_boundary.blur_radius,
                                opacity: start_boundary.opacity,
                                _pad: 0,
                            };
                            let composite_buffer = self.context.device.create_buffer_init(
                                &wgpu::util::BufferInitDescriptor {
                                    label: Some("filter_group_composite_buffer"),
                                    contents: bytemuck::cast_slice(std::slice::from_ref(
                                        &composite,
                                    )),
                                    usage: wgpu::BufferUsages::STORAGE,
                                },
                            );
                            let composite_bind_group = self.context.device.create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("filter_group_composite_bind_group"),
                                    layout: &self.pipelines.backdrop_filters_bind_group_layout,
                                    entries: &[wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::Buffer(
                                            wgpu::BufferBinding {
                                                buffer: &composite_buffer,
                                                offset: 0,
                                                size: None,
                                            },
                                        ),
                                    }],
                                },
                            );
                            let composite_texture_bind_group = self.context.device.create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("filter_group_texture_bind_group"),
                                    layout: &self.pipelines.backdrop_texture_bind_group_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: wgpu::BindingResource::TextureView(
                                                &self.group_views[depth],
                                            ),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::BindingResource::Sampler(
                                                &self.backdrop_blur_sampler,
                                            ),
                                        },
                                    ],
                                },
                            );

                            pass.set_pipeline(&self.pipelines.backdrop_filters_pipeline);
                            pass.set_bind_group(0, &self.pipelines.globals_bind_group, &[]);
                            pass.set_bind_group(1, &composite_bind_group, &[]);
                            pass.set_bind_group(2, &composite_texture_bind_group, &[]);
                            pass.draw(0..4, 0..1);
                        }
                    }
                    PrimitiveBatch::Underlines(underlines) => {
                        let count = underlines.len() as u32;
                        pass_state.set_pipeline(&mut pass, DrawPipelineId::Underlines, &self.pipelines.underlines_pipeline);
                        pass_state.set_bind_group(&mut pass, 0, BoundGroupId::Globals, &self.pipelines.globals_bind_group, &[]);
                        pass_state.set_bind_group(&mut pass, 1, BoundGroupId::LegacyBuffer(LegacyBuffer::Underlines), &underlines_bind_group, &[]);
                        pass_state.set_bind_group(&mut pass, 2, BoundGroupId::LayerTransform(0), &layer_transform_bind_group, &[0]);
                        pass.draw(
                            0..4,
                            underlines_first_instance..underlines_first_instance + count,
                        );
                        underlines_first_instance += count;
                        #[cfg(feature = "flamegraph")]
                        crate::record_draw_call(crate::DrawCallKind::Underlines, count);
                        #[cfg(feature = "flamegraph")]
                        if let Some(recorder) = deep_capture_recorder.as_mut() {
                            recorder.record_draw_call(
                                crate::DrawCallKind::Underlines,
                                "underlines",
                                current_pass_label,
                                0..4,
                                underlines_first_instance - count..underlines_first_instance,
                                2,
                                Some(crate::flamegraph::DeepCaptureBufferKind::Underlines),
                                None,
                                None,
                            );
                        }
                    }
                    PrimitiveBatch::Surfaces(surfaces) => {
                        log::trace!("Renderer: processing {} surface(s)", surfaces.len());
                        for surface in surfaces {
                            if let crate::SurfaceContent::Wgpu(surface_id) = &surface.content {
                                // Swap ready → display ONLY if the external renderer produced
                                // a new frame since we last composited this surface. This paint
                                // path runs every GPUI frame whether or not the producer rendered
                                // anything (the viewport re-arms request_animation_frame each
                                // frame), so an unconditional swap here would rotate `display` to
                                // a stale buffer whenever the producer skipped a frame — engine
                                // lock contention or a pending resize — and the canvas strobes.
                                // The gate holds the current display buffer until a real frame is
                                // ready. The fast-blit path (Path B) is already gated via
                                // redraw_pending, so it keeps using the unconditional swap.
                                let _swapped = self
                                    .context
                                    .surface_registry
                                    .swap_ready_display_if_new(*surface_id);

                                if let Some(view) =
                                    self.context.surface_registry.front_view(*surface_id)
                                {
                                    let params = SurfaceParams {
                                        bounds: Bounds {
                                            origin: [
                                                surface.bounds.origin.x.0,
                                                surface.bounds.origin.y.0,
                                            ],
                                            size: [
                                                surface.bounds.size.width.0,
                                                surface.bounds.size.height.0,
                                            ],
                                        },
                                        content_mask: Bounds {
                                            origin: [
                                                surface.content_mask.bounds.origin.x.0,
                                                surface.content_mask.bounds.origin.y.0,
                                            ],
                                            size: [
                                                surface.content_mask.bounds.size.width.0,
                                                surface.content_mask.bounds.size.height.0,
                                            ],
                                        },
                                    };

                                    // Cache bounds for fast surface blitting
                                    // Surface bounds are in ScaledPixels (f32), store as Pixels for caching
                                    self.surface_bounds_cache.lock().unwrap().insert(
                                        *surface_id,
                                        SurfaceBoundsEntry {
                                            screen_bounds: geometry::Bounds {
                                                origin: geometry::Point {
                                                    x: Pixels(surface.bounds.origin.x.0),
                                                    y: Pixels(surface.bounds.origin.y.0),
                                                },
                                                size: geometry::Size {
                                                    width: Pixels(surface.bounds.size.width.0),
                                                    height: Pixels(surface.bounds.size.height.0),
                                                },
                                            },
                                            content_mask: geometry::Bounds {
                                                origin: geometry::Point {
                                                    x: Pixels(
                                                        surface.content_mask.bounds.origin.x.0,
                                                    ),
                                                    y: Pixels(
                                                        surface.content_mask.bounds.origin.y.0,
                                                    ),
                                                },
                                                size: geometry::Size {
                                                    width: Pixels(
                                                        surface.content_mask.bounds.size.width.0,
                                                    ),
                                                    height: Pixels(
                                                        surface.content_mask.bounds.size.height.0,
                                                    ),
                                                },
                                            },
                                            layout_version: self
                                                .layout_version
                                                .load(Ordering::Acquire),
                                        },
                                    );

                                    let params_buffer = self.context.device.create_buffer_init(
                                        &wgpu::util::BufferInitDescriptor {
                                            label: Some("surface_params_buffer"),
                                            contents: bytemuck::bytes_of(&params),
                                            usage: wgpu::BufferUsages::UNIFORM,
                                        },
                                    );

                                    let surface_bind_group = self.context.device.create_bind_group(
                                        &wgpu::BindGroupDescriptor {
                                            label: Some("surface_bind_group"),
                                            layout: &self.pipelines.surfaces_bind_group_layout,
                                            entries: &[
                                                wgpu::BindGroupEntry {
                                                    binding: 0,
                                                    resource: wgpu::BindingResource::Buffer(
                                                        wgpu::BufferBinding {
                                                            buffer: &params_buffer,
                                                            offset: 0,
                                                            size: None,
                                                        },
                                                    ),
                                                },
                                                wgpu::BindGroupEntry {
                                                    binding: 1,
                                                    resource: wgpu::BindingResource::TextureView(
                                                        &view,
                                                    ),
                                                },
                                                wgpu::BindGroupEntry {
                                                    binding: 2,
                                                    resource: wgpu::BindingResource::Sampler(
                                                        &self.surface_sampler,
                                                    ),
                                                },
                                            ],
                                        },
                                    );

                                    pass.set_pipeline(&self.pipelines.surfaces_pipeline);
                                    pass.set_bind_group(0, &self.pipelines.globals_bind_group, &[]);
                                    pass.set_bind_group(1, &surface_bind_group, &[]);
                                    pass.draw(0..4, 0..1);
                                    // Per-surface groups are unique objects this
                                    // tracker does not model; drop all tracked
                                    // state so later draws rebind conservatively.
                                    pass_state.reset();
                                    #[cfg(feature = "flamegraph")]
                                    crate::record_draw_call(crate::DrawCallKind::Surfaces, 1);
                                    #[cfg(feature = "flamegraph")]
                                    if let Some(recorder) = deep_capture_recorder.as_mut() {
                                        recorder.record_draw_call(
                                            crate::DrawCallKind::Surfaces,
                                            "surfaces",
                                            current_pass_label,
                                            0..4,
                                            0..1,
                                            2,
                                            None,
                                            None,
                                            Some(surface_id.0),
                                        );
                                    }

                                    // CRITICAL: Keep view alive until after render pass ends
                                    // The bind_group holds a reference to it
                                    surface_views.push(view);
                                    surface_param_buffers.push(params_buffer);

                                    // Clear redraw pending AFTER we're done with the view
                                    // This prevents the external thread from triggering another compositor
                                    // pass while we're still using this view
                                    self.context
                                        .surface_registry
                                        .clear_redraw_pending(*surface_id);

                                    seen_surfaces.push(*surface_id);
                                }
                            } else if let crate::SurfaceContent::Layer(layer_id) = &surface.content
                            {
                                // #96: composite a texture-retained layer's
                                // persistent texture. The surface's bounds are
                                // the buffer extent (shifted by the buffered
                                // element's scroll); the content mask clips to
                                // the layer's visible rect, so margin content
                                // never paints outside the layer.
                                let Some(entry) = self.layer_textures.get_mut(layer_id) else {
                                    log::trace!(
                                        "layer texture for {layer_id:?} missing at composite; \
                                         waiting for the posted re-record"
                                    );
                                    continue;
                                };
                                entry.last_used_frame = self.layer_texture_frame;
                                let view = entry.view.clone();

                                let params = SurfaceParams {
                                    bounds: Bounds {
                                        origin: [
                                            surface.bounds.origin.x.0,
                                            surface.bounds.origin.y.0,
                                        ],
                                        size: [
                                            surface.bounds.size.width.0,
                                            surface.bounds.size.height.0,
                                        ],
                                    },
                                    content_mask: Bounds {
                                        origin: [
                                            surface.content_mask.bounds.origin.x.0,
                                            surface.content_mask.bounds.origin.y.0,
                                        ],
                                        size: [
                                            surface.content_mask.bounds.size.width.0,
                                            surface.content_mask.bounds.size.height.0,
                                        ],
                                    },
                                };

                                let params_buffer = self.context.device.create_buffer_init(
                                    &wgpu::util::BufferInitDescriptor {
                                        label: Some("layer_surface_params_buffer"),
                                        contents: bytemuck::bytes_of(&params),
                                        usage: wgpu::BufferUsages::UNIFORM,
                                    },
                                );

                                let surface_bind_group = self.context.device.create_bind_group(
                                    &wgpu::BindGroupDescriptor {
                                        label: Some("layer_surface_bind_group"),
                                        layout: &self.pipelines.surfaces_bind_group_layout,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: wgpu::BindingResource::Buffer(
                                                    wgpu::BufferBinding {
                                                        buffer: &params_buffer,
                                                        offset: 0,
                                                        size: None,
                                                    },
                                                ),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 1,
                                                resource: wgpu::BindingResource::TextureView(
                                                    &view,
                                                ),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 2,
                                                resource: wgpu::BindingResource::Sampler(
                                                    &self.surface_sampler,
                                                ),
                                            },
                                        ],
                                    },
                                );

                                pass.set_pipeline(&self.pipelines.surfaces_pipeline);
                                pass.set_bind_group(0, &self.pipelines.globals_bind_group, &[]);
                                pass.set_bind_group(1, &surface_bind_group, &[]);
                                pass.draw(0..4, 0..1);
                                pass_state.reset();
                                #[cfg(feature = "flamegraph")]
                                crate::record_draw_call(crate::DrawCallKind::Surfaces, 1);

                                // Keep the view alive until after the render pass ends.
                                surface_views.push(view);
                                surface_param_buffers.push(params_buffer);
                            }
                        }
                    }
                    PrimitiveBatch::Paths(paths) => {
                        let vertex_count: u32 = paths.iter().map(|p| p.vertices.len() as u32).sum();
                        if vertex_count > 0 {
                            pass_state.set_pipeline(&mut pass, DrawPipelineId::Paths, &self.pipelines.paths_pipeline);
                            pass_state.set_bind_group(&mut pass, 0, BoundGroupId::Globals, &self.pipelines.globals_bind_group, &[]);
                            pass_state.set_bind_group(&mut pass, 1, BoundGroupId::LegacyBuffer(LegacyBuffer::PathVertices), &paths_bind_group, &[]);
                            pass_state.set_bind_group(&mut pass, 2, BoundGroupId::LayerTransform(0), &layer_transform_bind_group, &[0]);
                            pass.draw(
                                paths_vertex_offset..paths_vertex_offset + vertex_count,
                                0..1,
                            );
                            paths_vertex_offset += vertex_count;
                            #[cfg(feature = "flamegraph")]
                            crate::record_draw_call(crate::DrawCallKind::Paths, paths.len() as u32);
                            #[cfg(feature = "flamegraph")]
                            if let Some(recorder) = deep_capture_recorder.as_mut() {
                                recorder.record_draw_call(
                                    crate::DrawCallKind::Paths,
                                    "paths",
                                    current_pass_label,
                                    paths_vertex_offset - vertex_count..paths_vertex_offset,
                                    0..1,
                                    2,
                                    Some(crate::flamegraph::DeepCaptureBufferKind::Paths),
                                    None,
                                    None,
                                );
                            }
                        }
                    }
                }
            }

            // The final span's merged stretch flushes here: nothing may keep
            // an open run across the pass end.
            if let Some(groups) = slab_groups.as_ref() {
                flush_open_slab_run(
                    &self.pipelines,
                    transform_slot_stride,
                    &mut pass,
                    groups,
                    &mut pass_state,
                    &mut open_slab_run,
                );
            }
        }

        // Blit persistent framebuffer to swapchain
        if let Some(ref persistent_framebuffer) = self.persistent_framebuffer {
            let extent = wgpu::Extent3d {
                width: self.surface_configuration.width,
                height: self.surface_configuration.height,
                depth_or_array_layers: 1,
            };
            command_encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: persistent_framebuffer,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &surface_texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                extent,
            );
        }

        // Close out the GpuSubmitPresent bracket and record the resolve +
        // resolve-to-staging copy for this frame's generation, before the
        // encoder is finished (resolve_query_set must be recorded on the same
        // encoder that wrote the timestamps).
        #[cfg(feature = "flamegraph")]
        if let Some(reserved) = &flamegraph_submit_present {
            command_encoder.write_timestamp(reserved.query_set(), reserved.end_index());
        }
        #[cfg(feature = "flamegraph")]
        {
            let mut guard = self.gpu_query_manager.lock();
            if let Some(manager) = guard.as_mut() {
                manager.finish_frame(&mut command_encoder);
            }
        }

        // On-demand GPU deep capture (issue #60): if this frame was armed for
        // recording, hand off from `DeepCaptureRecorder` to
        // `DeepCapturePendingReadback` now -- while `finish` still has a
        // chance to record any buffer-copy commands into `command_encoder`,
        // before it's finished below. `quads_buffer_ref`/etc. are the same
        // guards already held (further up in this function, for the
        // bind-group setup above) for the whole duration of `draw`, so this
        // reuses them rather than re-locking.
        #[cfg(feature = "flamegraph")]
        let deep_capture_pending = deep_capture_recorder.take().map(|recorder| {
            let buffers: [(crate::flamegraph::DeepCaptureBufferKind, &wgpu::Buffer); 7] = [
                (crate::flamegraph::DeepCaptureBufferKind::Quads, &quads_buffer_ref),
                (crate::flamegraph::DeepCaptureBufferKind::Shadows, &shadows_buffer_ref),
                (crate::flamegraph::DeepCaptureBufferKind::Underlines, &underlines_buffer_ref),
                (crate::flamegraph::DeepCaptureBufferKind::MonoSprites, &mono_sprites_buffer_ref),
                (crate::flamegraph::DeepCaptureBufferKind::PolySprites, &poly_sprites_buffer_ref),
                (
                    crate::flamegraph::DeepCaptureBufferKind::BackdropFilters,
                    &backdrop_filters_buffer_ref,
                ),
                (crate::flamegraph::DeepCaptureBufferKind::Paths, &paths_vertices_buffer_ref),
            ];
            recorder.finish(
                &self.context.device,
                &mut command_encoder,
                &buffers,
                &self.atlas,
                &self.context.surface_registry,
            )
        });

        log::trace!("Renderer::draw: submitting command buffer");
        self.context.queue.submit(Some(command_encoder.finish()));
        log::trace!("Renderer::draw: presenting surface");
        self.context.queue.present(surface_texture);

        // Start the async readback now that the resolve/copy commands above
        // have actually been submitted to the queue.
        #[cfg(feature = "flamegraph")]
        {
            let mut guard = self.gpu_query_manager.lock();
            if let Some(manager) = guard.as_mut() {
                manager.begin_readback();
            }
        }

        // Start this frame's deep-capture readback (if one was armed) now
        // that its buffer-copy commands, if any, have actually been
        // submitted, and hand it off to `self.deep_capture` so future
        // `draw()` calls poll it to completion. Overwrites (rather than
        // stacking on top of) any prior in-flight deep capture, but that
        // can't happen in practice: the arm step above only creates a new
        // recorder when `self.deep_capture` is `None`.
        #[cfg(feature = "flamegraph")]
        if let Some(mut pending) = deep_capture_pending {
            pending.begin_readback();
            *self.deep_capture.lock() = Some(pending);
        }

        log::trace!("Renderer::draw: frame complete");
    }

    /// Fast path: blit all visible surfaces in a single swapchain pass.
    /// Returns true if successful, false if compositor should run.
    pub fn blit_surfaces_direct(&self, pending_surfaces: &[SurfaceId]) -> bool {
        if pending_surfaces.is_empty() {
            return false;
        }

        let layout_version = self.layout_version.load(Ordering::Acquire);
        let mut visible_surfaces = {
            let cache = self.surface_bounds_cache.lock().unwrap();

            // Fast path is valid only when bounds are current.
            if cache
                .values()
                .any(|entry| entry.layout_version != layout_version)
            {
                return false;
            }

            // Every pending surface must be currently visible with cached bounds.
            if pending_surfaces
                .iter()
                .any(|surface_id| !cache.contains_key(surface_id))
            {
                return false;
            }

            cache
                .iter()
                .map(|(surface_id, entry)| (*surface_id, entry.screen_bounds, entry.content_mask))
                .collect::<Vec<_>>()
        };

        if visible_surfaces.is_empty() {
            return false;
        }

        // Keep deterministic ordering.
        visible_surfaces.sort_unstable_by_key(|(surface_id, _, _)| surface_id.0);

        // Flip ready -> display for surfaces that actually rendered new frames.
        for surface_id in pending_surfaces {
            let _ = self
                .context
                .surface_registry
                .swap_ready_display(&self.context.device, *surface_id);
        }

        // Acquire swapchain (handle retryable surface errors the same as regular draw).
        let surface_texture = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(t)
            | CurrentSurfaceTexture::Suboptimal(t) => t,
            CurrentSurfaceTexture::Outdated
            | CurrentSurfaceTexture::Lost
            | CurrentSurfaceTexture::Validation => {
                self.reconfigure_surface();
                match self.surface.get_current_texture() {
                    CurrentSurfaceTexture::Success(t)
                    | CurrentSurfaceTexture::Suboptimal(t) => t,
                    other => {
                        log::warn!(
                            "Fast blit failed to acquire swapchain after reconfigure: {:?}",
                            other
                        );
                        return false;
                    }
                }
            }
            CurrentSurfaceTexture::Timeout => {
                log::warn!("Fast blit failed: swapchain acquire timed out");
                return false;
            }
            CurrentSurfaceTexture::Occluded => {
                log::warn!("Fast blit failed: swapchain acquire occluded");
                return false;
            }
        };

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fast_surface_blit"),
                });
        let swapchain_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        {
            // Only gets a timestamp span when a WgpuRenderer::draw-initiated
            // frame is still in its `Recording` state (see
            // `GpuQueryManager::reserve_pair` and `reserve_gpu_timestamps`'s
            // doc comment) — blit_surfaces_direct can also run outside any
            // open frame (the no-compositor fast path bypasses `draw()`
            // entirely), in which case this is `None` and the pass simply
            // isn't captured this round.
            #[cfg(feature = "flamegraph")]
            let flamegraph_fast_blit_pass = self.reserve_gpu_timestamps(
                crate::SpanName::Static("fast_surface_blit_pass"),
                crate::GpuPassKind::FastSurfaceBlit,
            );

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fast_surface_blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swapchain_view,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Preserve existing swapchain content
                        store: wgpu::StoreOp::Store,
                    },
                    resolve_target: None,
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                #[cfg(feature = "flamegraph")]
                timestamp_writes: flamegraph_fast_blit_pass.as_ref().map(|reserved| reserved.writes()),
                #[cfg(not(feature = "flamegraph"))]
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.pipelines.surfaces_pipeline);
            pass.set_bind_group(0, &self.pipelines.globals_bind_group, &[]);

            // Keep views and params buffers alive until pass ends (bind groups reference them).
            let mut surface_views = Vec::new();
            let mut surface_param_buffers = Vec::new();

            for (surface_id, screen_bounds, content_mask) in &visible_surfaces {
                let Some(view) = self.context.surface_registry.front_view(*surface_id) else {
                    return false;
                };

                let params = SurfaceParams {
                    bounds: Bounds {
                        origin: [screen_bounds.origin.x.0, screen_bounds.origin.y.0],
                        size: [screen_bounds.size.width.0, screen_bounds.size.height.0],
                    },
                    content_mask: Bounds {
                        origin: [content_mask.origin.x.0, content_mask.origin.y.0],
                        size: [content_mask.size.width.0, content_mask.size.height.0],
                    },
                };

                let params_buffer =
                    self.context
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("fast_blit_surface_params_buffer"),
                            contents: bytemuck::bytes_of(&params),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                let surface_bind_group =
                    self.context
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("fast_blit_surface_bind_group"),
                            layout: &self.pipelines.surfaces_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &params_buffer,
                                        offset: 0,
                                        size: None,
                                    }),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(&view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(&self.surface_sampler),
                                },
                            ],
                        });

                pass.set_bind_group(1, &surface_bind_group, &[]);
                pass.draw(0..4, 0..1);
                surface_views.push(view);
                surface_param_buffers.push(params_buffer);
            }
        }

        self.context.queue.submit(Some(encoder.finish()));
        self.context.queue.present(surface_texture);

        // Clear redraw flags only for surfaces that presented fresh frames.
        for surface_id in pending_surfaces {
            self.context
                .surface_registry
                .clear_redraw_pending(*surface_id);
        }

        true
    }

    /// Get list of surfaces that have pending redraws
    pub fn get_pending_surfaces(&self) -> Option<Vec<SurfaceId>> {
        let pending = self.context.surface_registry.get_pending_surfaces();
        if pending.is_empty() {
            None
        } else {
            Some(pending)
        }
    }

    /// Present without running compositor (fast blit already updated swapchain)
    pub fn present_framebuffer_only(&self) {
        // NOTE: Fast blit already presented to swapchain, so this is a no-op
        // When we implement persistent framebuffer properly, this will blit framebuffer → swapchain
        log::trace!("Present framebuffer only (no compositor) - fast blit already presented");
    }

    pub fn update_drawable_size(&mut self, size: geometry::Size<DevicePixels>) {
        self.surface_configuration.width = size.width.0 as u32;
        self.surface_configuration.height = size.height.0 as u32;
        self.reconfigure_surface();

        // Recreate persistent framebuffer at new size
        let persistent_framebuffer = self
            .context
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("persistent_framebuffer"),
                size: wgpu::Extent3d {
                    width: self.surface_configuration.width,
                    height: self.surface_configuration.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_configuration.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });

        let persistent_framebuffer_view =
            persistent_framebuffer.create_view(&wgpu::TextureViewDescriptor::default());

        self.persistent_framebuffer = Some(persistent_framebuffer);
        self.persistent_framebuffer_view = Some(persistent_framebuffer_view);

        // Recreate backdrop blur capture texture at the new size so that
        // copy_texture_to_texture doesn't silently skip due to a size mismatch.
        let backdrop_blur_texture = self
            .context
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("backdrop_blur_texture"),
                size: wgpu::Extent3d {
                    width: self.surface_configuration.width,
                    height: self.surface_configuration.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_configuration.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
        let backdrop_blur_texture_view =
            backdrop_blur_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.backdrop_blur_texture = Some(backdrop_blur_texture);
        self.backdrop_blur_texture_view = Some(backdrop_blur_texture_view);

        // Recreate the content-filter group textures at the new size so they stay
        // pixel-aligned with the surface (group composites sample using
        // `pixel_position / globals.viewport_size` UVs, just like backdrop blur).
        let (group_textures, group_views) = create_filter_group_textures(
            &self.context.device,
            self.surface_configuration.width,
            self.surface_configuration.height,
            self.surface_configuration.format,
        );
        self.group_textures = group_textures;
        self.group_views = group_views;

        // Layer textures (#96) are sized to their layer's buffer extent, not
        // the surface, so they survive a resize — but their content was
        // rasterized against the old scale/viewport, and the composite's NDC
        // mapping changed. Drop them all; the re-record requests make each
        // texture-retained layer re-bake on its next composite.
        let dropped_keys: Vec<crate::LayerKey> = self
            .layer_textures
            .drain()
            .map(|(_, entry)| entry.key)
            .collect();
        if !dropped_keys.is_empty() {
            log::trace!(
                "dropped {} layer textures for resize; requesting re-records",
                dropped_keys.len()
            );
            self.slab_registry.request_rerecord(dropped_keys);
        }

        // Invalidate bounds cache - all surface bounds are now stale
        self.layout_version.fetch_add(1, Ordering::Release);
        self.surface_bounds_cache.lock().unwrap().clear();
    }

    pub fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.atlas.clone()
    }

    pub fn gpu_specs(&self) -> GpuSpecs {
        let info = self.context.adapter.get_info();
        GpuSpecs {
            is_software_emulated: info.device_type == wgpu::DeviceType::Cpu,
            device_name: info.name,
            driver_name: info.driver,
            driver_info: info.driver_info,
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.surface_configuration.alpha_mode = if transparent {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else {
            #[cfg(target_os = "linux")]
            {
                wgpu::CompositeAlphaMode::Inherit
            }
            #[cfg(not(target_os = "linux"))]
            {
                wgpu::CompositeAlphaMode::Opaque
            }
        };
        self.reconfigure_surface();
    }

    pub fn viewport_size(&self) -> geometry::Size<DevicePixels> {
        geometry::Size {
            width: DevicePixels(self.surface_configuration.width as i32),
            height: DevicePixels(self.surface_configuration.height as i32),
        }
    }

    /// On-demand GPU memory snapshot for this renderer (Phase 3 of the
    /// profiling epic, issue #59): mostly summing sizes that already exist
    /// on already-owned wgpu resources, no new tracking required.
    #[cfg(feature = "flamegraph")]
    pub(crate) fn gpu_memory_snapshot(&self) -> crate::GpuMemorySnapshot {
        crate::GpuMemorySnapshot {
            fixed_buffer_bytes: self.context.fixed_buffer_memory_usage(),
            atlas_bytes: self.atlas.memory_usage(),
            surface_registry_bytes: self.context.surface_registry.memory_usage(),
            swapchain_bytes: self.swapchain_memory_usage(),
        }
    }

    /// The live `wgpu::Device`/`Queue` backing this renderer, for callers
    /// that want to drive GPU work outside the normal frame path -- e.g.
    /// `flamegraph_replay::render_deep_capture_step` (Phase 6 of the
    /// profiling epic, issue #62), so a deep-capture replay preview runs
    /// against the app's real device instead of spinning up a second,
    /// separate headless one on every call. `Device`/`Queue` are cheap
    /// `Clone` handles (wgpu itself reference-counts the underlying
    /// resources), so returning owned clones here is the idiomatic wgpu
    /// pattern, not a meaningful cost.
    #[cfg(feature = "flamegraph")]
    pub(crate) fn gpu_device_and_queue(&self) -> (wgpu::Device, wgpu::Queue) {
        (self.context.device.clone(), self.context.queue.clone())
    }

    /// Best-effort swapchain memory estimate from `surface_configuration`'s
    /// dimensions/format. wgpu doesn't expose the presentation engine's
    /// actual backing image count, so `desired_maximum_frame_latency` (the
    /// one buffering-depth signal WGPUI itself configures) stands in for it.
    #[cfg(feature = "flamegraph")]
    fn swapchain_memory_usage(&self) -> u64 {
        let bytes_per_texel = super::render_context::texel_size(self.surface_configuration.format);
        let image_count = self.surface_configuration.desired_maximum_frame_latency.max(1) as u64;
        (self.surface_configuration.width as u64)
            * (self.surface_configuration.height as u64)
            * bytes_per_texel
            * image_count
    }
}

/// Issue one merged slab run's instanced draw with full bind state.
///
/// Free-standing (rather than a `WgpuRenderer` method) so the GPU-tier tests
/// can drive the exact production draw function against a headless device.
/// Every call rebinds everything, which is what makes it the naive baseline;
/// production goes through [`flush_slab_run_with_state`] so redundant binds
/// are skipped instead.
#[cfg(test)]
fn flush_slab_run(
    pipelines: &WgpuPipelines,
    transform_slot_stride: u64,
    pass: &mut wgpu::RenderPass<'_>,
    slabs: &crate::platform::cross::slab::LayerSlabs,
    groups: &SlabDrawGroups,
    transform_slot: u32,
    run: &SlabPendingRun,
) {
    let mut untracked = PassBindState::default();
    flush_slab_run_with_state(
        pipelines,
        transform_slot_stride,
        pass,
        slabs,
        groups,
        transform_slot,
        run,
        &mut untracked,
    );
}

/// [`flush_slab_run`] with bind-state tracking: pipeline and bind-group sets
/// that would repeat what the pass already holds are skipped. Skipping is
/// pixel-neutral because tracked ids match only when the bound resource,
/// layout slot, and dynamic offsets are all identical.
fn flush_slab_run_with_state(
    pipelines: &WgpuPipelines,
    transform_slot_stride: u64,
    pass: &mut wgpu::RenderPass<'_>,
    slabs: &crate::platform::cross::slab::LayerSlabs,
    groups: &SlabDrawGroups,
    transform_slot: u32,
    run: &SlabPendingRun,
    state: &mut PassBindState,
) {
    profiling::scope!("wgpui: flush slab runs");
    let dynamic_offsets = [(transform_slot as u64 * transform_slot_stride) as u32];
    let transform_id = BoundGroupId::LayerTransform(dynamic_offsets[0]);
    let range_base = slabs.slab(run.kind).base + run.start;
    match run.kind {
        SlabKind::Quads => {
            state.set_pipeline(pass, DrawPipelineId::Quads, &pipelines.quads_pipeline);
            state.set_bind_group(pass, 0, BoundGroupId::Globals, &pipelines.globals_bind_group, &[]);
            state.set_bind_group(pass, 1, BoundGroupId::SlabStorage(SlabKind::Quads), &groups.quads, &[]);
            state.set_bind_group(pass, 2, transform_id, &groups.layer_transform, &dynamic_offsets);
            pass.draw(0..4, range_base..range_base + run.count);
        }
        SlabKind::Shadows => {
            state.set_pipeline(pass, DrawPipelineId::Shadows, &pipelines.shadows_pipeline);
            state.set_bind_group(pass, 0, BoundGroupId::Globals, &pipelines.globals_bind_group, &[]);
            state.set_bind_group(pass, 1, BoundGroupId::SlabStorage(SlabKind::Shadows), &groups.shadows, &[]);
            state.set_bind_group(pass, 2, transform_id, &groups.layer_transform, &dynamic_offsets);
            pass.draw(0..4, range_base..range_base + run.count);
        }
            SlabKind::Underlines => {
                state.set_pipeline(pass, DrawPipelineId::Underlines, &pipelines.underlines_pipeline);
                state.set_bind_group(pass, 0, BoundGroupId::Globals, &pipelines.globals_bind_group, &[]);
                state.set_bind_group(pass, 1, BoundGroupId::SlabStorage(SlabKind::Underlines), &groups.underlines, &[]);
                state.set_bind_group(pass, 2, transform_id, &groups.layer_transform, &dynamic_offsets);
                pass.draw(0..4, range_base..range_base + run.count);
            }
            SlabKind::MonoSprites => {
                let Some(texture_id) = run.texture_id else {
                    debug_assert!(false, "sprite runs carry a texture id");
                    return;
                };
                let Some(texture_group) = groups.sprite_textures.get(&(texture_id.index, texture_id.kind))
                else {
                    debug_assert!(false, "sprite textures validated before drawing");
                    return;
                };
                state.set_pipeline(pass, DrawPipelineId::MonoSprites, &pipelines.mono_sprites_pipeline);
                state.set_bind_group(pass, 0, BoundGroupId::Globals, &pipelines.globals_bind_group, &[]);
                state.set_bind_group(pass, 1, BoundGroupId::ColorAdjustments, &pipelines.color_adjustments_bind_group, &[]);
                state.set_bind_group(
                    pass,
                    2,
                    BoundGroupId::SpriteTexture(texture_id.index, texture_id.kind),
                    texture_group,
                    &[],
                );
                state.set_bind_group(pass, 3, BoundGroupId::SlabStorage(SlabKind::MonoSprites), &groups.mono_sprites, &[]);
                state.set_bind_group(pass, 4, transform_id, &groups.layer_transform, &dynamic_offsets);
                pass.draw(0..4, range_base..range_base + run.count);
            }
            SlabKind::PolySprites => {
                let Some(texture_id) = run.texture_id else {
                    debug_assert!(false, "sprite runs carry a texture id");
                    return;
                };
                let Some(texture_group) = groups.sprite_textures.get(&(texture_id.index, texture_id.kind))
                else {
                    debug_assert!(false, "sprite textures validated before drawing");
                    return;
                };
                state.set_pipeline(pass, DrawPipelineId::PolySprites, &pipelines.poly_sprites_pipeline);
                state.set_bind_group(pass, 0, BoundGroupId::Globals, &pipelines.globals_bind_group, &[]);
                state.set_bind_group(
                    pass,
                    1,
                    BoundGroupId::SpriteTexture(texture_id.index, texture_id.kind),
                    texture_group,
                    &[],
                );
                state.set_bind_group(pass, 2, BoundGroupId::SlabStorage(SlabKind::PolySprites), &groups.poly_sprites, &[]);
                state.set_bind_group(pass, 3, transform_id, &groups.layer_transform, &dynamic_offsets);
                pass.draw(0..4, range_base..range_base + run.count);
            }
            // Path runs address vertices, not instances: the layer's vertex
            // block sits at its Paths range inside the shared stream.
            SlabKind::Paths => {
                let base = slabs.slab(SlabKind::Paths).base;
                state.set_pipeline(pass, DrawPipelineId::Paths, &pipelines.paths_pipeline);
                state.set_bind_group(pass, 0, BoundGroupId::Globals, &pipelines.globals_bind_group, &[]);
                state.set_bind_group(pass, 1, BoundGroupId::SlabStorage(SlabKind::Paths), &groups.paths_vertices, &[]);
                state.set_bind_group(pass, 2, transform_id, &groups.layer_transform, &dynamic_offsets);
                pass.draw(base + run.start..base + run.start + run.count, 0..1);
            }
        }
        crate::render_stats::add(slab_gpu::COUNTER_DRAW_CALLS, 1);
        #[cfg(feature = "flamegraph")]
        crate::record_draw_call(flamegraph_kind(run.kind), run.count);
    }

/// Flush an open cross-span slab stretch, if any, with the shared bind-state
/// tracker. Called before any non-continuing draw and at end of pass.
fn flush_open_slab_run(
    pipelines: &WgpuPipelines,
    transform_slot_stride: u64,
    pass: &mut wgpu::RenderPass<'_>,
    groups: &SlabDrawGroups,
    state: &mut PassBindState,
    open: &mut Option<OpenSlabRun>,
) {
    if let Some(open) = open.take() {
        let pending = open.as_pending();
        flush_slab_run_with_state(
            pipelines,
            transform_slot_stride,
            pass,
            &open.slabs,
            groups,
            open.transform_slot,
            &pending,
            state,
        );
    }
}

#[cfg(feature = "flamegraph")]
fn flamegraph_kind(kind: SlabKind) -> crate::DrawCallKind {
    match kind {
        SlabKind::Quads => crate::DrawCallKind::Quads,
        SlabKind::Shadows => crate::DrawCallKind::Shadows,
        SlabKind::Paths => crate::DrawCallKind::Paths,
        SlabKind::Underlines => crate::DrawCallKind::Underlines,
        SlabKind::MonoSprites => crate::DrawCallKind::MonoSprites,
        SlabKind::PolySprites => crate::DrawCallKind::PolySprites,
    }
}


impl Drop for WgpuRenderer {
    fn drop(&mut self) {
        // SAFETY: This is the only Drop impl and `surface` has not been dropped yet.
        // We take it manually so we can drop it inside catch_unwind, suppressing the Vulkan
        // panic that occurs when a SurfaceTexture's Arc still holds a swapchain semaphore
        // reference at the time the surface is destroyed (e.g. window closed mid-frame).
        let surface = unsafe { ManuallyDrop::take(&mut self.surface) };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            drop(surface);
        }));
    }
}

#[cfg(test)]
#[path = "renderer_slab_tests.rs"]
mod slab_tests;
