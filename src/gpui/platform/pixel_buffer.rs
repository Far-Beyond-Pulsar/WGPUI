//! A CPU-side RGBA8 image captured from a rendered frame, plus comparison and
//! golden-diff helpers used by the rendering audit kit.

use std::path::Path;

use anyhow::{Context as _, Result};

/// An RGBA8 image (8 bits per channel, byte order R, G, B, A) read back from the
/// GPU. Produced by the offscreen audit harness and by
/// [`crate::Window::read_back_pixels`].
///
/// The bytes are already gamma-encoded (the renderer writes sRGB values into a
/// non-sRGB target), so compare them directly against expected sRGB byte values;
/// do not apply a second sRGB decode.
#[derive(Clone, Debug)]
pub struct PixelBuffer {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major, tightly packed RGBA8 bytes (`width * height * 4` long).
    pub pixels: Vec<u8>,
}

impl PixelBuffer {
    /// Construct from raw, tightly packed RGBA8 bytes.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        Self {
            width,
            height,
            pixels,
        }
    }

    /// The RGBA bytes of the pixel at `(x, y)`, or `[0; 4]` if out of bounds.
    pub fn pixel_at(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height {
            return [0; 4];
        }
        let index = ((y * self.width + x) * 4) as usize;
        match self.pixels.get(index..index + 4) {
            Some(slice) => [slice[0], slice[1], slice[2], slice[3]],
            None => [0; 4],
        }
    }

    /// Whether two buffers have identical dimensions.
    pub fn same_dimensions(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height
    }

    /// The largest absolute per-channel difference between any two corresponding
    /// pixels, or `None` if dimensions differ. `0` means pixel-identical.
    pub fn max_channel_diff(&self, other: &Self) -> Option<u8> {
        if !self.same_dimensions(other) {
            return None;
        }
        Some(
            self.pixels
                .iter()
                .zip(&other.pixels)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0),
        )
    }

    /// The number of pixels whose maximum per-channel difference exceeds
    /// `tolerance`, or `None` if dimensions differ. Use for golden comparison
    /// with an anti-aliasing / driver tolerance.
    pub fn pixels_exceeding(&self, other: &Self, tolerance: u8) -> Option<usize> {
        if !self.same_dimensions(other) {
            return None;
        }
        let count = self
            .pixels
            .chunks_exact(4)
            .zip(other.pixels.chunks_exact(4))
            .filter(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .any(|(x, y)| x.abs_diff(*y) > tolerance)
            })
            .count();
        Some(count)
    }

    /// Save as a PNG, for inspection or as a golden reference.
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let buffer = image::RgbaImage::from_raw(self.width, self.height, self.pixels.clone())
            .context("pixel buffer length does not match its dimensions")?;
        buffer
            .save(path)
            .with_context(|| format!("failed to save PNG to {}", path.display()))?;
        Ok(())
    }

    /// Load a PNG (e.g. a golden reference) as an RGBA8 buffer.
    pub fn load_png(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let image = image::open(path)
            .with_context(|| format!("failed to open PNG {}", path.display()))?
            .to_rgba8();
        Ok(Self::new(image.width(), image.height(), image.into_raw()))
    }
}
