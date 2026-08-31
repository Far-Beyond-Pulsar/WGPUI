// The instanced polychrome-sprite pipeline: one image per instance, pulling its
// slot out of the `PolySprite` arena through §5.3's indirection buffer and
// reading its colour bitmap out of one atlas page.
//
// Structurally this is `mono_sprites.wgsl` over a colour page. The instance
// addressing is identical and deliberately so — `visible[slot.base + instance]`,
// the same `SlotBase` uniform under both `FirstInstance` encodings, the same
// four-vertex triangle strip, the same page filter — because §5.3's claim is
// about a *fixed* draw sequence per (layer, kind) slot, and an image layer is a
// slot like any other.
//
// Three things are genuinely different from the glyph shader, and each of them
// is a decision rather than an inevitability.
//
// # 1. The quad's size and the tile's size are two numbers, not one
//
// A glyph blits its tile one texel to one pixel: the atlas holds the raster at
// the size it will be drawn, and `atlas_size` is therefore both the tile extent
// and the quad extent. An image does not — layout decides how big the picture
// is, and the decode decides how big the bitmap is, and those agree only when
// the image happens to be drawn at its natural size. So the sprite carries both,
// and the fragment shader maps the quad's local coordinate through the ratio.
//
// # 2. `textureLoad`, not a sampler — and what that costs, stated plainly
//
// The legacy `poly_sprites.wgsl` uses `textureSample` with a linear filter.
// This does not, for the reason `mono_sprites.wgsl` records: a `textureLoad` at
// an integer address is exactly comparable against the CPU-side page bytes, and
// "the pixel on screen *is* the texel in the atlas" is a statement a test can
// assert as equality rather than as a tolerance. At the natural size — which is
// where the byte-exact proof lives, and the case `object_fit: None` and a
// correctly-sized layout box both produce — the two are identical anyway,
// because a linear sample at a texel centre returns that texel.
//
// Where they differ is a *scaled* image: legacy interpolates and this takes the
// nearest texel, so a downscaled photograph is visibly harsher here. That is a
// real fidelity gap and it is named in docs/phase-6.2-results.md rather than
// left for someone to discover. It is also a self-contained change — a sampler,
// a bind-group entry, and normalised coordinates — held back only because
// making it now would have cost this phase the exactness its gate is built on.
//
// # 3. The corner radius is a signed-distance cut, matching legacy
//
// `quad_sdf`, transcribed from `src/platform/cross/shaders/poly_sprites.wgsl`
// including its `saturate(0.5 - distance)` edge term, so a rounded avatar's
// antialiased rim is the legacy rim and not an approximation of it.

struct Globals {
    // Framebuffer size in pixels.
    viewport: vec2<f32>,
    padding: vec2<f32>,
};

// Four scalars rather than one word plus a `vec3<u32>`, for the reason
// `quads.wgsl` records: WGSL aligns a `vec3<u32>` to 16 bytes, so the obvious
// spelling would need a 32-byte `min_binding_size` for one useful word.
struct SlotBase {
    base: u32,
    padding_0: u32,
    translation: vec2<f32>,
    clip_origin: vec2<f32>,
    clip_size: vec2<f32>,
};

// Which atlas page the bound texture is. See `mono_sprites.wgsl`'s header for
// why the page is a filter rather than a lookup; the argument is identical and
// is not repeated.
struct AtlasPage {
    index: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
};

// 48 bytes, matching `wgpui_core::patch::primitive::PolySprite::SLOT_STRIDE` and
// the field order `PolySprite::encode` writes.
//
// Every member is naturally aligned where the encoder puts it: four
// `vec2<f32>` at 0/8/16/24 (alignment 8), then four scalars at 32/36/40/44.
// Unlike `GlyphSlot` there is no field that WGSL would want to over-align, so
// this struct can be spelled the obvious way.
struct SpriteSlot {
    origin: vec2<f32>,
    size: vec2<f32>,
    atlas_origin: vec2<f32>,
    atlas_size: vec2<f32>,
    corner_radius: f32,
    opacity: f32,
    grayscale: u32,
    atlas_tile: u32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> sprites: array<SpriteSlot>;
@group(0) @binding(2) var<storage, read> visible: array<u32>;
@group(1) @binding(0) var<uniform> slot: SlotBase;
@group(2) @binding(0) var<uniform> page: AtlasPage;
@group(2) @binding(1) var atlas: texture_2d<f32>;
@group(2) @binding(2) var atlas_sampler: sampler;

// `wgpui_core::indirect::UNUSED_INSTANCE`.
const UNUSED_INSTANCE: u32 = 0xffffffffu;
// `wgpui_core::patch::primitive::AtlasTileId::NONE`.
const NO_TILE: u32 = 0xffffffffu;
// `AtlasTileId::SLOT_BITS`.
const TILE_SLOT_BITS: u32 = 24u;

// The legacy shader's own constants, transcribed: Rec. 709 luma weights.
const GRAYSCALE_FACTORS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // Position within the drawn rectangle, in pixels — not in tile texels, so
    // the fragment stage can do both the corner-radius distance (which is about
    // the rectangle) and the texel lookup (which is about the tile) from it.
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) arena_index: u32,
};

// Every corner at one clip-space point, so the two triangles have zero area and
// produce no fragments.
fn degenerate() -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
    out.local = vec2<f32>(0.0, 0.0);
    out.arena_index = 0u;
    return out;
}

@vertex
fn vertex_main(
    @builtin(vertex_index) corner: u32,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    let arena_index = visible[slot.base + instance];
    if (arena_index == UNUSED_INSTANCE) {
        // Reachable only if an argument record and the indirection buffer
        // disagree, which is a bug — but a bug that must not draw sprite 0 many
        // times over.
        return degenerate();
    }

    let sprite = sprites[arena_index];
    // An image that has not decoded yet, or whose tile the atlas refused. It
    // holds its slab slot on purpose (`patch/primitive.rs`) and draws nothing.
    if (sprite.atlas_tile == NO_TILE) {
        return degenerate();
    }
    // Another page's sprite: that page's pass draws it.
    if ((sprite.atlas_tile >> TILE_SLOT_BITS) != page.index) {
        return degenerate();
    }

    var out: VertexOutput;
    // Four corners of a triangle strip: (0,0) (1,0) (0,1) (1,1).
    let unit = vec2<f32>(f32(corner & 1u), f32((corner >> 1u) & 1u));
    let point = sprite.origin + unit * sprite.size + slot.translation;
    out.position = vec4<f32>(
        point.x / globals.viewport.x * 2.0 - 1.0,
        1.0 - point.y / globals.viewport.y * 2.0,
        0.0,
        1.0,
    );
    out.local = unit * sprite.size;
    out.arena_index = arena_index;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if slot.clip_size.x >= 0.0 {
        let clip_max = slot.clip_origin + slot.clip_size;
        if in.position.x < slot.clip_origin.x || in.position.y < slot.clip_origin.y
            || in.position.x >= clip_max.x || in.position.y >= clip_max.y {
            discard;
        }
    }
    let sprite = sprites[in.arena_index];

    // Map the position within the drawn rectangle onto the tile. At the natural
    // size this is the identity and the load is a 1:1 blit; otherwise it is
    // nearest-neighbour. Guarded against a zero-sized rectangle, which is
    // representable (`PolySprite::ZERO`) and would otherwise divide by zero.
    let extent = max(sprite.size, vec2<f32>(1.0, 1.0));
    let scaled = in.local / extent * sprite.atlas_size;
    // Clamped rather than trusted: interpolation at the quad's far edge can land
    // exactly on `atlas_size`, and a `textureLoad` outside the tile would read a
    // neighbouring image's texels instead of this one's.
    let last = max(sprite.atlas_size - vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0));
    let inside = clamp(floor(scaled), vec2<f32>(0.0, 0.0), last);
    var color: vec4<f32>;
    if (all(abs(sprite.size - sprite.atlas_size) < vec2<f32>(0.001, 0.001))) {
        color = textureLoad(atlas, vec2<i32>(sprite.atlas_origin + inside), 0);
    } else {
        let dimensions = vec2<f32>(textureDimensions(atlas));
        let tile_min = (sprite.atlas_origin + vec2<f32>(0.5, 0.5)) / dimensions;
        let tile_max = (sprite.atlas_origin + sprite.atlas_size - vec2<f32>(0.5, 0.5)) / dimensions;
        let coordinate = clamp(
            (sprite.atlas_origin + scaled + vec2<f32>(0.5, 0.5)) / dimensions,
            tile_min,
            tile_max,
        );
        color = textureSampleLevel(atlas, atlas_sampler, coordinate, 0.0);
    }

    if (sprite.grayscale != 0u) {
        let luma = dot(color.rgb, GRAYSCALE_FACTORS);
        color = vec4<f32>(vec3<f32>(luma), color.a);
    }

    // The legacy rounded-rectangle cut, including its half-pixel edge term, so a
    // rounded avatar's rim is the legacy rim. `quad_sdf` is evaluated in the
    // rectangle's own space, which is what `local` is.
    let coverage = saturate(0.5 - rounded_rect_distance(in.local, sprite.size, sprite.corner_radius));
    let alpha = color.a * sprite.opacity * coverage;
    if (alpha <= 0.0) {
        discard;
    }
    return vec4<f32>(color.rgb, alpha);
}

// `quad_sdf` from `src/platform/cross/shaders/poly_sprites.wgsl`, specialised to
// one uniform radius because `PolySprite` carries one — see that type's doc for
// why four radii were not added to two primitive kinds for this phase.
//
// `point` is relative to the rectangle's top-left, so the centre is `size / 2`.
fn rounded_rect_distance(point: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
    let half_size = size / 2.0;
    let center_to_point = point - half_size;
    let corner_to_point = abs(center_to_point) - half_size;
    let corner_center_to_point = corner_to_point + radius;
    if (radius == 0.0) {
        return max(corner_center_to_point.x, corner_center_to_point.y);
    }
    let inset = length(max(vec2<f32>(0.0), corner_center_to_point))
        + min(0.0, max(corner_center_to_point.x, corner_center_to_point.y));
    return inset - radius;
}
