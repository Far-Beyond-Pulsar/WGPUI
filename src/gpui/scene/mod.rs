mod batch;
mod batch_counter;
mod batch_iterator;
mod batch_iterator_steps;
mod chunk;
mod finish;
mod paint;
mod path;
mod primitive;
mod state;
mod transform;

#[cfg(test)]
mod tests;

pub(crate) use batch::PrimitiveBatch;
pub(crate) use chunk::{ChangedRanges, SceneChunk};
pub use path::Path;
pub(crate) use path::PathId;
pub use primitive::BorderStyle;
pub(crate) use primitive::{
    BackdropBlur,
    DrawOrder,
    MonochromeSprite,
    PaintOperation,
    PaintSurface,
    PolychromeSprite,
    Primitive,
    PrimitiveKind,
    Quad,
    Shadow,
    SurfaceContent,
    Underline,
};
pub(crate) use state::Scene;
pub use transform::TransformationMatrix;
