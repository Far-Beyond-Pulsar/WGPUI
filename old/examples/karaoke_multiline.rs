use std::time::Duration;

use gpui::*;

/// Multi-line karaoke with per-character timing
/// Using public domain lyrics: "Twinkle Twinkle Little Star"
struct MultilineKaraoke {
    lines: Vec<KaraokeLine>,
    current_line: usize,
    line_progress: f32,
}

struct KaraokeLine {
    text: SharedString,
    char_timings: Vec<f32>, // Duration in seconds for each character
    total_duration: f32,
}

impl KaraokeLine {
    fn new(text: impl Into<SharedString>, char_timings: Vec<f32>) -> Self {
        let total_duration = char_timings.iter().sum();
        Self {
            text: text.into(),
            char_timings,
            total_duration,
        }
    }

    /// Get the progress (0.0 to 1.0) through the text given elapsed time
    fn progress_at_time(&self, elapsed: f32) -> f32 {
        if elapsed >= self.total_duration {
            return 1.0;
        }

        let mut accumulated = 0.0;
        let mut chars_completed = 0;

        for (i, &duration) in self.char_timings.iter().enumerate() {
            if accumulated + duration > elapsed {
                // We're in the middle of this character
                let char_progress = (elapsed - accumulated) / duration;
                let total_chars = self.char_timings.len() as f32;
                return (i as f32 + char_progress) / total_chars;
            }
            accumulated += duration;
            chars_completed = i + 1;
        }

        chars_completed as f32 / self.char_timings.len() as f32
    }
}

impl MultilineKaraoke {
    fn new(cx: &mut Context<Self>) -> Self {
        // Public domain lyrics: "Twinkle Twinkle Little Star"
        // Variable timing per character to match singing rhythm
        let lines = vec![
            KaraokeLine::new(
                "Twinkle, twinkle, little star,",
                vec![0.15, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.15, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.15, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.15, 0.08, 0.08, 0.08, 0.08, 0.15]
            ),
            KaraokeLine::new(
                "How I wonder what you are!",
                vec![0.15, 0.08, 0.08, 0.10, 0.10, 0.12, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.10, 0.08, 0.08, 0.08, 0.08, 0.10, 0.08, 0.08, 0.08, 0.10, 0.08, 0.08, 0.08, 0.12]
            ),
            KaraokeLine::new(
                "Up above the world so high,",
                vec![0.15, 0.08, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.10, 0.08, 0.08, 0.08, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.10, 0.08, 0.08, 0.08, 0.08, 0.15]
            ),
            KaraokeLine::new(
                "Like a diamond in the sky.",
                vec![0.15, 0.08, 0.08, 0.08, 0.10, 0.08, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.10, 0.08, 0.08, 0.10, 0.08, 0.08, 0.08, 0.10, 0.08, 0.08, 0.08, 0.15]
            ),
        ];

        let karaoke = Self {
            lines,
            current_line: 0,
            line_progress: 0.0,
        };

        // Start animation
        cx.spawn(async move |this, cx| {
            loop {
                for line_idx in 0..4 {
                    let duration = this.update(cx, |this, _| {
                        this.current_line = line_idx;
                        this.lines[line_idx].total_duration
                    }).ok().unwrap_or(3.0);

                    let steps = (duration * 60.0) as u32; // 60 FPS
                    for i in 0..=steps {
                        let elapsed = (i as f32 / steps as f32) * duration;
                        this.update(cx, |this, cx| {
                            this.line_progress = this.lines[this.current_line].progress_at_time(elapsed);
                            cx.notify();
                        }).ok();
                        Timer::after(Duration::from_millis((1000.0 / 60.0) as u64)).await;
                    }

                    // Pause between lines
                    Timer::after(Duration::from_millis(500)).await;
                }

                // Pause before restarting
                Timer::after(Duration::from_secs(2)).await;

                this.update(cx, |this, cx| {
                    this.current_line = 0;
                    this.line_progress = 0.0;
                    cx.notify();
                }).ok();
            }
        }).detach();

        karaoke
    }
}

impl Render for MultilineKaraoke {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let colors = [
            (rgb(0xff1744), rgb(0x444444)), // Red to gray
            (rgb(0x00e676), rgb(0x444444)), // Green to gray
            (rgb(0x00b0ff), rgb(0x444444)), // Blue to gray
            (rgb(0xffd600), rgb(0x444444)), // Yellow to gray
        ];

        div()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .size_full()
            .bg(rgb(0x0a0a0a))
            .gap_6()
            .child(
                div()
                    .text_size(px(32.0))
                    .text_color(rgb(0x888888))
                    .font_weight(FontWeight::BOLD)
                    .child("Twinkle Twinkle Little Star"),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_4()
                    .children(
                        self.lines.iter().enumerate().map(|(idx, line)| {
                            let (active_color, inactive_color) = colors[idx % colors.len()];
                            let is_current = idx == self.current_line;
                            let progress = if is_current {
                                self.line_progress
                            } else if idx < self.current_line {
                                1.0 // Already sung
                            } else {
                                0.0 // Not yet sung
                            };

                            let opacity = if is_current {
                                1.0
                            } else if idx < self.current_line {
                                0.5
                            } else {
                                0.3
                            };

                            div()
                                .text_size(px(if is_current { 42.0 } else { 36.0 }))
                                .font_weight(if is_current {
                                    FontWeight::BOLD
                                } else {
                                    FontWeight::NORMAL
                                })
                                .text_gradient_horizontal(
                                    linear_color_stop(active_color, progress.max(0.01) - 0.01),
                                    linear_color_stop(inactive_color, progress.max(0.01) + 0.01),
                                )
                                .with_opacity(opacity)
                                .child(line.text.clone())
                        }),
                    ),
            )
            .child(
                div()
                    .text_size(px(14.0))
                    .text_color(rgb(0x666666))
                    .child(format!(
                        "Line {}/4 • Progress: {:.0}%",
                        self.current_line + 1,
                        self.line_progress * 100.0
                    )),
            )
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1000.0), px(600.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|cx| MultilineKaraoke::new(cx)),
        )
        .expect("Failed to open window");
    });
}
