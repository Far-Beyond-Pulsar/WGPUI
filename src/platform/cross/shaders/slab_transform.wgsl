// Per-layer transform uniform, injected ahead of every shader whose pipeline
// can draw spliced layer-slab content. The bind group sits at the highest
// position in each pipeline layout; `{SLAB_TRANSFORM_GROUP}` is replaced by
// the renderer before module creation, since WGSL has no include mechanism.
//
// Packed instance data carries coordinates relative to its layer's origin;
// the vertex stage adds `layer_transform.translate` to restore window space.
// Wherever a fragment stage compares interpolated framebuffer position
// against untranslated instance geometry, it must subtract the same value —
// exactly once — via `layer_world_position`.
struct LayerTransform {
    translate: vec2<f32>,
    _pad0: vec2<f32>,
    _pad1: vec4<f32>,
    _pad2: vec4<f32>,
    _pad3: vec4<f32>,
}

@group({SLAB_TRANSFORM_GROUP}) @binding(0) var<uniform> layer_transform: LayerTransform;

fn layer_world_position(framebuffer_position: vec2<f32>) -> vec2<f32> {
    return framebuffer_position - layer_transform.translate;
}
