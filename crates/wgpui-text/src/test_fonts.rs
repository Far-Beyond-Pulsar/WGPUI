//! One embedded face, and nothing else, for tests that need to assert about
//! actual glyph pixels.
//!
//! `docs/phase-5-results.md` §11 names the gap this fills: Phase 5's shaping
//! tests "assert glyph counts, monotonic advance, run grouping, and cache
//! behaviour — never a specific glyph id or advance width, because the tests
//! shape against whatever fonts the machine has", and it says embedding a face
//! would fix it. A rasterisation test cannot make that trade at all — "these are
//! the right pixels" is meaningless against an unknown font — so Phase 5.5
//! embeds the face Phase 5 said it would take to do this properly.
//!
//! IBM Plex Sans Regular, the same file the legacy backend already bundles for
//! WASM (`src/platform/cross/text_system.rs`, where there are no system fonts),
//! under the SIL Open Font License already vendored beside it. It is loaded into
//! a database containing *only* it, so a resolution can only land on one face
//! and a test's expectations cannot silently start describing a different one.

use crate::shaping::TextShaper;
use cosmic_text::{FontSystem, fontdb};

/// The embedded face's family name, as its `name` table reports it.
pub const FAMILY: &str = "IBM Plex Sans";

/// The face itself.
pub const REGULAR: &[u8] = include_bytes!("../../../assets/fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf");

/// A font database holding the embedded face and nothing else.
pub fn font_system() -> FontSystem {
    let mut database = fontdb::Database::new();
    database.load_font_data(REGULAR.to_vec());
    FontSystem::new_with_locale_and_db("en-US".to_owned(), database)
}

/// A shaper over [`font_system`].
pub fn shaper() -> TextShaper {
    TextShaper::with_font_system(font_system())
}
