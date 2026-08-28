// The instanced monochrome-sprite pipeline: one glyph per instance, pulling its
// slot out of the `GlyphRun` arena through §5.3's indirection buffer and
// sampling the coverage mask out of one atlas page.
//
// Structurally this is `quads.wgsl` with a texture. The instance addressing is
// identical and deliberately so — `visible[slot.base + instance]`, the same
// `SlotBase` uniform under both `FirstInstance` encodings, the same four-vertex
// triangle strip — because §5.3's claim is about a *fixed* draw sequence per
// (layer, kind) slot, and a text layer is a slot like any other.
//
// # Why the page index is a filter rather than a lookup
//
// A glyph names its atlas tile as a packed `(page, slot)` word
// (`AtlasTileId`), and the page decides which texture holds its texels. A bind
// group cannot change inside a draw call, so a slot whose glyphs span two pages
// cannot be one draw. The alternatives are a binding array of every page (a
// device feature, and one WebGPU does not guarantee) or drawing the slot once
// per page with the glyphs of other pages dropped. This shader takes the second:
// `page.index` is bound alongside the texture, and a glyph belonging to another
// page collapses to a degenerate triangle strip exactly as an unused instance
// does. The CPU still never learns an instance count — it issues the same
// indirect record once per live page.
//
// In the common case there is exactly one monochrome page, so this costs one
// extra comparison per glyph and nothing else.
//
// # `textureLoad`, not a sampler
//
// The atlas holds `SUBPIXEL_VARIANTS_X` horizontal rasters of every glyph
// (`wgpui_text::patch`), which is the legacy design's way of carrying sub-pixel
// positioning in the *raster* rather than in the *sample*. A glyph quad is
// therefore meant to blit its tile one texel to one pixel, and a 1:1 blit needs
// no filtering and no normalised coordinates — it needs the texel at an integer
// address, which is what `textureLoad` is. It also makes the result exactly
// comparable against the CPU-side page bytes, which is what
// `tests/glyph_sprite_draw.rs` compares.

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
    padding_1: u32,
    padding_2: u32,
};

// Which atlas page the bound texture is. See this file's header.
struct AtlasPage {
    index: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
};

// 48 bytes, matching `wgpui_core::patch::primitive::GlyphRun::SLOT_STRIDE` and
// the field order `GlyphRun::encode` writes.
//
// The colour is four scalars rather than a `vec4<f32>` because it starts at byte
// 28, and WGSL would align a `vec4<f32>` member to 32 — which would silently
// read the wrong bytes for every field after it. Four `f32`s align to 4 and land
// where the encoder put them.
struct GlyphSlot {
    position: vec2<f32>,
    atlas_origin: vec2<f32>,
    atlas_size: vec2<f32>,
    glyph_id: u32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    color_a: f32,
    atlas_tile: u32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> glyphs: array<GlyphSlot>;
@group(0) @binding(2) var<storage, read> visible: array<u32>;
@group(1) @binding(0) var<uniform> slot: SlotBase;
@group(2) @binding(0) var<uniform> page: AtlasPage;
@group(2) @binding(1) var atlas: texture_2d<f32>;

// `wgpui_core::indirect::UNUSED_INSTANCE`.
const UNUSED_INSTANCE: u32 = 0xffffffffu;
// `wgpui_core::patch::primitive::AtlasTileId::NONE`.
const NO_TILE: u32 = 0xffffffffu;
// `AtlasTileId::SLOT_BITS`.
const TILE_SLOT_BITS: u32 = 24u;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
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
        // disagree, which is a bug — but a bug that must not draw glyph 0 many
        // times over.
        return degenerate();
    }

    let glyph = glyphs[arena_index];
    // Whitespace shapes to a positioned glyph with a real advance and no
    // coverage. It holds its slab slot on purpose (`patch/primitive.rs`) and
    // draws nothing.
    if (glyph.atlas_tile == NO_TILE) {
        return degenerate();
    }
    // Another page's glyph: that page's pass draws it.
    if ((glyph.atlas_tile >> TILE_SLOT_BITS) != page.index) {
        return degenerate();
    }

    var out: VertexOutput;
    // Four corners of a triangle strip: (0,0) (1,0) (0,1) (1,1).
    let unit = vec2<f32>(f32(corner & 1u), f32((corner >> 1u) & 1u));
    let point = glyph.position + unit * glyph.atlas_size;
    out.position = vec4<f32>(
        point.x / globals.viewport.x * 2.0 - 1.0,
        1.0 - point.y / globals.viewport.y * 2.0,
        0.0,
        1.0,
    );
    out.local = unit * glyph.atlas_size;
    out.arena_index = arena_index;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let glyph = glyphs[in.arena_index];
    // Clamped rather than trusted: interpolation at the quad's far edge can land
    // exactly on `atlas_size`, and a `textureLoad` outside the tile would read a
    // neighbouring glyph's texels instead of this one's.
    let last = max(glyph.atlas_size - vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0));
    let inside = clamp(floor(in.local), vec2<f32>(0.0, 0.0), last);
    let coverage = textureLoad(atlas, vec2<i32>(glyph.atlas_origin + inside), 0).r;
    if (coverage <= 0.0) {
        discard;
    }
    return vec4<f32>(
        glyph.color_r,
        glyph.color_g,
        glyph.color_b,
        glyph.color_a * coverage,
    );
}
