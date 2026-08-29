//! The whole glyph path, end to end, and the only part of it that needs a
//! device.
//!
//! Shape real text with `wgpui-text`, rasterise every glyph through the ported
//! `SwashCache` path, pack the bitmaps into `wgpui-wgpu`'s atlas, upload the
//! pages to real `wgpu::Texture`s, read the textures back, and compare them to
//! the CPU-side pages byte for byte.
//!
//! Everything before the upload is device-free and is tested where it lives.
//! What can only be checked here is that the copy is *correct* — that a page's
//! texels arrive at the right coordinates with the right row alignment — and
//! that is the whole reason this file opens an adapter. Per Phase 3's rule, a
//! missing adapter is reported plainly and never allowed to look like coverage
//! that ran.

use wgpui_core::scene::atlas::{
    AtlasKind, GlyphRasterKey, GlyphTileSource, ImageRasterKey, RasterizedGlyph, RasterizedImage,
};
use wgpui_wgpu::render::atlas::{AtlasTileSource, GlyphAtlas};
use wgpui_wgpu::render::atlas_upload::{AtlasTextures, texture_format};
use wgpui_wgpu::render::device::context_or_report;
use wgpui_text::patch::{RunPlacement, glyph_runs};
use wgpui_text::raster::GlyphRasterizer;
use wgpui_text::shaping::{FontRun, SharedString, font};
use wgpui_text::test_fonts;

/// Read a whole atlas page texture back into a tightly-packed buffer.
///
/// Undoes the `COPY_BYTES_PER_ROW_ALIGNMENT` padding on the way out, so the
/// result can be compared directly with `GlyphAtlas::page_texels` — which is the
/// point: the comparison has to be against what the CPU side holds, not against
/// a re-derivation of what it should hold.
fn read_page_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    kind: AtlasKind,
) -> Vec<u8> {
    let width = texture.width() as usize;
    let height = texture.height() as usize;
    let bytes_per_pixel = kind.bytes_per_pixel() as usize;
    let unpadded = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded = unpadded.div_ceil(align) * align;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("atlas page readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("atlas page readback"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: texture.width(),
            height: texture.height(),
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device must finish the copy");
    receiver
        .recv()
        .expect("the map callback must run")
        .expect("the map must succeed");

    let tight = {
        let view = slice.get_mapped_range().expect("mapped range");
        let mut tight = Vec::with_capacity(unpadded * height);
        for row in 0..height {
            let start = row * padded;
            tight.extend_from_slice(&view[start..start + unpadded]);
        }
        tight
    };
    staging.unmap();
    tight
}

fn key(glyph: u32, kind: AtlasKind) -> GlyphRasterKey {
    GlyphRasterKey {
        font: 0,
        glyph,
        font_size_bits: 16.0f32.to_bits(),
        subpixel: [0, 0],
        scale_factor_bits: 1.0f32.to_bits(),
        kind,
    }
}

/// A bitmap whose every texel is distinct, so a misplaced row or a wrong stride
/// shows up as a mismatch rather than as an identical-looking block.
fn ramp(width: u32, height: u32, kind: AtlasKind, seed: u8) -> RasterizedGlyph {
    let bytes = (width * height * kind.bytes_per_pixel()) as usize;
    RasterizedGlyph {
        size: [width, height],
        kind,
        bearing: [0.0, 0.0],
        texels: (0..bytes)
            .map(|index| seed.wrapping_add((index % 251) as u8))
            .collect(),
    }
}

#[test]
fn an_uploaded_page_reads_back_exactly_as_the_cpu_side_holds_it() {
    let Some(context) = context_or_report("an_uploaded_page_reads_back_exactly") else {
        return;
    };

    // Small pages, so a readback is kilobytes and so the multi-page path is
    // reached rather than hoped for.
    let mut atlas = GlyphAtlas::new(128);
    let mut textures = AtlasTextures::for_atlas(&atlas);

    for glyph in 0..12u32 {
        let size = 8 + glyph * 3;
        atlas
            .get_or_insert_raster(
                key(glyph, AtlasKind::Monochrome),
                &ramp(size, size / 2 + 1, AtlasKind::Monochrome, glyph as u8),
            )
            .expect("a 128px page holds a dozen small rasters");
    }
    // And a colour page, whose stride is four times as wide — the case a
    // monochrome-only test would never catch.
    for glyph in 100..104u32 {
        atlas
            .get_or_insert_raster(
                key(glyph, AtlasKind::Polychrome),
                &ramp(6, 9, AtlasKind::Polychrome, glyph as u8),
            )
            .expect("a colour raster");
    }

    let report = textures.sync(&context.device, &context.queue, &mut atlas);
    println!("upload: {report:?}");
    assert_eq!(report.rectangles, 16);
    assert_eq!(report.pages_created, 2, "one monochrome page and one colour");
    assert_eq!(report.skipped, 0);
    assert!(!atlas.has_pending_uploads(), "sync drains what it uploads");

    for page in atlas.page_indices() {
        let kind = atlas.page_kind(page).expect("a live page has a kind");
        let texture = textures.texture(page).expect("every live page has a texture");
        assert_eq!(texture.format(), texture_format(kind));
        let read_back = read_page_back(&context.device, &context.queue, texture, kind);
        let expected = atlas.page_texels(page).expect("a live page has texels");
        assert_eq!(
            read_back.len(),
            expected.len(),
            "page {page} read back the wrong number of bytes"
        );
        assert!(
            read_back == expected,
            "page {page} ({kind:?}) differs from the CPU side at texel {}",
            read_back
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .unwrap_or(0)
        );
        assert!(
            read_back.iter().any(|texel| *texel != 0),
            "page {page} read back entirely blank, which would make the comparison vacuous"
        );
    }
}

/// Phase 6.2's layer-2 proof, at the level this file works at: an image frame
/// inserted through the *image* entry point reaches a real `Rgba8Unorm` texture
/// with its bytes unchanged.
///
/// Deliberately in this file rather than a new one. The upload machinery is
/// already kind-generic — `AtlasTextures::sync` was written against
/// `AtlasKind`, not against glyphs — and the honest way to show that a second
/// producer needed no upload-side work is to put its case beside the first
/// producer's and use the same readback, not to write a parallel one that could
/// diverge.
#[test]
fn an_image_frame_reaches_a_colour_texture_with_its_bytes_unchanged() {
    let Some(context) = context_or_report("an_image_frame_reaches_a_colour_texture") else {
        return;
    };
    let mut atlas = GlyphAtlas::new(128);
    let mut textures = AtlasTextures::for_atlas(&atlas);

    // 37 texels wide: 148 unpadded bytes per row, which is not a multiple of
    // `COPY_BYTES_PER_ROW_ALIGNMENT`, so the copy has to pad and the readback
    // has to unpad. A power-of-two width would let both bugs cancel.
    let frame = RasterizedImage {
        size: [37, 11],
        texels: (0..37 * 11 * 4)
            .map(|index: u32| (index % 251) as u8)
            .collect(),
    };
    let placement = atlas
        .get_or_insert_image(
            ImageRasterKey {
                source: 7,
                frame_index: 0,
                scale_factor_bits: 1.0f32.to_bits(),
            },
            &frame,
        )
        .expect("a 37x11 frame fits a 128px page");

    let report = textures.sync(&context.device, &context.queue, &mut atlas);
    assert_eq!(report.rectangles, 1);
    assert_eq!(report.pages_created, 1);
    assert_eq!(report.skipped, 0);

    let page = placement.tile.page().expect("a live tile has a page");
    assert_eq!(atlas.page_kind(page), Some(AtlasKind::Polychrome));
    let texture = textures.texture(page).expect("the page has a texture");
    assert_eq!(texture.format(), texture_format(AtlasKind::Polychrome));

    let read_back = read_page_back(
        &context.device,
        &context.queue,
        texture,
        AtlasKind::Polychrome,
    );
    // The whole page matches the CPU side, and — the part that matters — the
    // rectangle the tile occupies holds the decoded bytes in row order.
    let expected_page = atlas.page_texels(page).expect("a live page has texels");
    assert!(read_back == expected_page, "the colour page diverged from the CPU side");

    let origin = [placement.origin[0] as usize, placement.origin[1] as usize];
    let stride = 128 * 4;
    for row in 0..11usize {
        let start = (origin[1] + row) * stride + origin[0] * 4;
        let source = row * 37 * 4;
        assert_eq!(
            read_back.get(start..start + 37 * 4),
            frame.texels.get(source..source + 37 * 4),
            "row {row} of the uploaded frame is not the row that was decoded"
        );
    }
}

#[test]
fn a_second_sync_uploads_only_what_changed() {
    let Some(context) = context_or_report("a_second_sync_uploads_only_what_changed") else {
        return;
    };
    let mut atlas = GlyphAtlas::new(128);
    let mut textures = AtlasTextures::for_atlas(&atlas);

    atlas
        .get_or_insert_raster(
            key(1, AtlasKind::Monochrome),
            &ramp(10, 10, AtlasKind::Monochrome, 1),
        )
        .expect("allocate");
    let first = textures.sync(&context.device, &context.queue, &mut atlas);
    assert_eq!(first.rectangles, 1);

    // The same glyph again: resident, so nothing is written and nothing is
    // uploaded. This is the steady-state claim — a window whose text did not
    // change costs the atlas nothing.
    atlas
        .get_or_insert_raster(
            key(1, AtlasKind::Monochrome),
            &ramp(10, 10, AtlasKind::Monochrome, 1),
        )
        .expect("resident");
    let second = textures.sync(&context.device, &context.queue, &mut atlas);
    assert_eq!(second, Default::default(), "a resident atlas uploads nothing");

    atlas
        .get_or_insert_raster(
            key(2, AtlasKind::Monochrome),
            &ramp(10, 10, AtlasKind::Monochrome, 2),
        )
        .expect("allocate");
    let third = textures.sync(&context.device, &context.queue, &mut atlas);
    assert_eq!(third.rectangles, 1, "one new glyph is one copy");
    assert_eq!(third.pages_created, 0, "and no new page");
}

#[test]
fn destroying_a_page_drops_its_texture_on_the_next_sync() {
    let Some(context) = context_or_report("destroying_a_page_drops_its_texture") else {
        return;
    };
    let mut atlas = GlyphAtlas::new(128);
    let mut textures = AtlasTextures::for_atlas(&atlas);
    atlas
        .get_or_insert_raster(
            key(1, AtlasKind::Monochrome),
            &ramp(10, 10, AtlasKind::Monochrome, 1),
        )
        .expect("allocate");
    textures.sync(&context.device, &context.queue, &mut atlas);
    assert_eq!(textures.page_count(), 1);

    assert!(atlas.destroy_page(0));
    let report = textures.sync(&context.device, &context.queue, &mut atlas);
    assert_eq!(report.pages_destroyed, 1);
    assert_eq!(textures.page_count(), 0);
    assert!(textures.view(0).is_none());
}

/// Shape → rasterise → pack → upload → read back, with nothing synthetic in it.
///
/// This is the sentence §9's risk row says was not true of `2.0`: "nothing
/// anywhere in `2.0` yet rasterises a glyph's font outline into the pixels that
/// allocated tile is supposed to hold."
#[test]
fn real_text_reaches_a_real_texture() {
    let Some(context) = context_or_report("real_text_reaches_a_real_texture") else {
        return;
    };

    let mut shaper = test_fonts::shaper();
    let font_id = shaper
        .resolve_font(&font(test_fonts::FAMILY))
        .expect("the embedded face resolves");
    let text = SharedString::from("Hello, world — glyphs with actual pixels.");
    let line = shaper
        .shape_line(&text, 24.0, &[FontRun::new(text.len(), font_id)])
        .expect("shaping must succeed");
    assert!(line.glyph_count() > 20);

    let mut atlas = GlyphAtlas::new(256);
    let mut rasterizer = GlyphRasterizer::new();
    let (runs, stats) = {
        let mut source = AtlasTileSource::new(&mut atlas, |key| {
            rasterizer.rasterize(&mut shaper, key).ok()
        });
        // A resident glyph is answered without rasterising, so the same 'l' in
        // "Hello" costs one raster; that is `AtlasTileSource`'s job and it is
        // checked in its own module. Here it just has to not break.
        let _ = source.tile_for(GlyphRasterKey {
            glyph: u32::MAX,
            ..key(0, AtlasKind::Monochrome)
        });
        glyph_runs(&line, RunPlacement::default(), &mut source)
    };

    println!(
        "shaped {} glyphs into {} runs: {} tiles, {} blanks; atlas {:?}; rasteriser {:?}",
        stats.glyphs,
        stats.runs,
        stats.tiles_referenced,
        stats.blank_glyphs,
        atlas.stats(),
        rasterizer.stats(),
    );
    assert_eq!(stats.glyphs, line.glyph_count());
    assert!(
        stats.tiles_referenced > 20,
        "most of a Latin line has ink: {} of {}",
        stats.tiles_referenced,
        stats.glyphs
    );
    assert!(stats.blank_glyphs > 0, "the spaces must still hold their slots");
    assert!(atlas.stats().allocations > 10);
    assert!(
        atlas.stats().cache_hits > 0,
        "a repeated letter must be answered from the atlas, not rasterised again"
    );
    assert!(!runs.is_empty());

    let mut textures = AtlasTextures::for_atlas(&atlas);
    let report = textures.sync(&context.device, &context.queue, &mut atlas);
    println!("upload: {report:?}");
    assert_eq!(report.rectangles as u64, atlas.stats().allocations);
    assert_eq!(report.skipped, 0);

    for page in atlas.page_indices() {
        let kind = atlas.page_kind(page).expect("a kind");
        let texture = textures.texture(page).expect("a texture");
        let read_back = read_page_back(&context.device, &context.queue, texture, kind);
        assert!(
            read_back == atlas.page_texels(page).expect("texels"),
            "page {page} does not match the CPU side"
        );
        assert!(
            read_back.iter().any(|coverage| *coverage > 0),
            "a page of real text with no coverage in it is not real text"
        );
    }

    // And the glyph slots point at texels that actually exist: every non-blank
    // glyph's tile must read back with ink in it. This is the assertion that
    // would have failed at every point in 2.0 before this phase.
    let mut inked = 0usize;
    for run in &runs {
        for glyph in &run.glyphs {
            if glyph.atlas_tile.is_none() {
                continue;
            }
            let texels = atlas
                .tile_texels(wgpui_wgpu::render::atlas::TilePlacement {
                    tile: glyph.atlas_tile,
                    kind: AtlasKind::Monochrome,
                    origin: glyph.atlas_origin,
                    size: glyph.atlas_size,
                    bearing: [0.0, 0.0],
                })
                .expect("a referenced tile is resident");
            if texels.iter().any(|coverage| *coverage > 0) {
                inked += 1;
            }
        }
    }
    assert_eq!(
        inked, stats.tiles_referenced,
        "every glyph that claims a tile must find ink in it"
    );
}
