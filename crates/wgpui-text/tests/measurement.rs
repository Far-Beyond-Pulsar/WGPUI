use wgpui_text::shaping::{Font, FontRun, SharedString, TextShaper};

#[test]
fn measurement_reuses_the_shaping_cache_and_reports_real_metrics() {
    let mut shaper = TextShaper::new();
    let font = shaper
        .resolve_font(&Font::default())
        .expect("fallback font");
    let text = SharedString::from("GPU-native text");
    let runs = [FontRun::new(text.len(), font)];

    let first = shaper
        .measure_line(&text, 16.0, &runs)
        .expect("text measures");
    let before = shaper.stats();
    let second = shaper
        .measure_line(&text, 16.0, &runs)
        .expect("cached text measures");

    assert_eq!(first, second);
    assert!(first.width > 0.0);
    assert!(first.ascent > 0.0);
    assert!(first.descent >= 0.0);
    assert_eq!(first.byte_length, text.len());
    assert_eq!(shaper.stats().lines_shaped, before.lines_shaped);
    assert_eq!(shaper.stats().cache_hits, before.cache_hits + 1);
}
