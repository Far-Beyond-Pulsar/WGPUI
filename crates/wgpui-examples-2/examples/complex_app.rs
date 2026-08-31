//! A deliberately dense native WGPUI application built against the current
//! direct API.

use std::cell::Cell;
use std::rc::Rc;

use wgpui::{ApplicationError, NativeApplication, Styled, WindowOptions, div, rgb};

fn main() -> Result<(), ApplicationError> {
    let selected = Rc::new(Cell::new(0_u32));
    let selected_for_button = Rc::clone(&selected);
    let inspected = Rc::new(Cell::new(false));
    let inspected_for_button = Rc::clone(&inspected);
    let hovered_control = Rc::new(Cell::new(0_u8));
    let hovered_control_for_first = Rc::clone(&hovered_control);
    let hovered_control_for_second = Rc::clone(&hovered_control);
    let scroll_offset = Rc::new(Cell::new(0.0_f32));
    let scroll_offset_for_handler = Rc::clone(&scroll_offset);

    NativeApplication::new(WindowOptions::default(), move |window| {
        window.performance_debug().set_tile_refresh_flash(
            wgpui::TileRefreshFlash::enabled()
                .with_tile_size(256.0, 256.0)
                .with_color([1.0, 0.0, 1.0, 0.35]),
        );
        let _ = window.interaction();
        let selected = selected.get();
        let inspected = inspected.get();
        let hovered_control = hovered_control.get();
        let scroll_offset = scroll_offset.get();
        let button_color = if selected == 0 {
            rgb(0x2f6fed)
        } else {
            rgb(0x2459bd)
        };
        let first_button_color = if hovered_control == 1 {
            rgb(0x4f8dff)
        } else {
            button_color
        };
        let second_button_color = if hovered_control == 2 {
            rgb(0x273653)
        } else {
            rgb(0x171d29)
        };

        div()
            .id("application")
            .size_full()
            .min_h(0.0)
            .p_6()
            .flex()
            .flex_col()
            .gap_4()
            .bg(rgb(0x10141c))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(div().text_2xl().text_color(rgb(0xf4f7ff)).child("Command Center"))
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(rgb(0x9ca9c2))
                                    .child("Retained GPU-native application overview"),
                            ),
                    )
                    .child(
                        div()
                            .px_3()
                            .py_1()
                            .rounded_lg()
                            .bg(rgb(0x183b2e))
                            .text_sm()
                            .text_color(rgb(0x71e0ad))
                            .child("ONLINE"),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_3()
                    .child(stat_card("Frame time", "1.8 ms", rgb(0x20365f)))
                    .child(stat_card("GPU passes", "12", rgb(0x3d2d5d)))
                    .child(stat_card("Resident tiles", "248", rgb(0x244b43))),
            )
            .child(
                div()
                    .flex()
                    .gap_4()
                    .flex_1()
                    .min_h(0.0)
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_3()
                            .flex_1()
                            .min_h(0.0)
                            .p_4()
                            .rounded_lg()
                            .border_1()
                            .border_color(rgb(0x293348))
                            .bg(rgb(0x171d29))
                            .id("activity-scroll")
                            .boundary()
                            .overflow_y_scroll()
                            .scroll_offset([0.0, -scroll_offset])
                            .on_scroll({
                                let scroll_offset = Rc::clone(&scroll_offset_for_handler);
                                move |event, _, _| {
                                    scroll_offset.set(
                                        (scroll_offset.get() - event.delta[1])
                                            .clamp(0.0, 520.0),
                                    );
                                }
                            })
                            .child(div().text_lg().text_color(rgb(0xf4f7ff)).child("Recent activity"))
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(0xff73d9ff))
                                    .child("Scroll this panel to inspect retained activity content"),
                            )
                            .child(activity_row("GPU scene compacted", "just now", rgb(0x54d69b)))
                            .child(activity_row("Atlas page resident", "12 sec ago", rgb(0x5ca8ff)))
                            .child(activity_row("Tile boundary crossed", "48 sec ago", rgb(0xe6b85c)))
                            .child(activity_row("Surface synchronized", "2 min ago", rgb(0xb18cff)))
                            .child(activity_row("Indirect args rebuilt", "3 min ago", rgb(0xf28c68)))
                            .child(activity_row("Atlas upload delta", "4 min ago", rgb(0xd18cff)))
                            .child(activity_row("Occlusion pass complete", "5 min ago", rgb(0x62d4e8)))
                            .child(activity_row("Input region updated", "6 min ago", rgb(0xffc857)))
                            .child(activity_row("Surface presented", "7 min ago", rgb(0x8fd694)))
                            .child(activity_row("Retained node reused", "8 min ago", rgb(0x9ca9ff)))
                            .child(activity_row("Glyph page compacted", "9 min ago", rgb(0xe78ac3))),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_3()
                            .w(250.0)
                            .p_4()
                            .rounded_lg()
                            .border_1()
                            .border_color(rgb(0x293348))
                            .bg(rgb(0x171d29))
                            .child(div().text_lg().text_color(rgb(0xf4f7ff)).child("Controls"))
                            .child(
                                div()
                                    .h(44.0)
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded_lg()
                                    .bg(first_button_color)
                                    .text_color(rgb(0xffffff))
                            .child(format!("Rebuild visible tiles ({selected})"))
                                    .on_click({
                                        let selected = Rc::clone(&selected_for_button);
                                        move |_, _, _| {
                                            selected.set(selected.get().wrapping_add(1));
                                        }
                                    })
                                    .on_hover({
                                        let hovered_control = Rc::clone(&hovered_control_for_first);
                                        move |is_hovered, _, _| {
                                            hovered_control.set(if is_hovered { 1 } else { 0 });
                                        }
                                    }),
                            )
                            .child(
                                div()
                                    .h(44.0)
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded_lg()
                                    .border_1()
                                    .border_color(rgb(0x34415d))
                                    .bg(second_button_color)
                                    .text_color(rgb(0xb8c4dc))
                                    .child(if inspected {
                                        "Scene inspected"
                                    } else {
                                        "Inspect retained scene"
                                    })
                                    .on_click({
                                        let inspected = Rc::clone(&inspected_for_button);
                                        move |_, _, _| {
                                            inspected.set(!inspected.get());
                                        }
                                    })
                                    .on_hover({
                                        let hovered_control = Rc::clone(&hovered_control_for_second);
                                        move |is_hovered, _, _| {
                                            hovered_control.set(if is_hovered { 2 } else { 0 });
                                        }
                                    }),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(0x7f8ba5))
                                    .child("Actions update retained state without rebuilding unchanged content."),
                            )
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .p_3()
                    .rounded_lg()
                    .bg(rgb(0x1c2738))
                    .child(div().text_sm().text_color(rgb(0xcbd6ec)).child("All systems nominal"))
                    .child(div().text_xs().text_color(rgb(0x8291ad)).child("WGPUI 2.0 native backend")),
            )
    })
    .run()
}

fn stat_card(label: &'static str, value: &'static str, color: wgpui::Rgba) -> wgpui::Div {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .flex_1()
        .p_4()
        .rounded_lg()
        .bg(color)
        .child(div().text_xs().text_color(rgb(0xc0cbe0)).child(label))
        .child(div().text_2xl().text_color(rgb(0xffffff)).child(value))
}

fn activity_row(label: &'static str, time: &'static str, color: wgpui::Rgba) -> wgpui::Div {
    div()
        .flex()
        .flex_shrink_0()
        .items_center()
        .gap_2()
        .p_2()
        .rounded_md()
        .child(div().w(8.0).h(8.0).rounded_lg().bg(color))
        .child(
            div()
                .flex()
                .flex_col()
                .flex_1()
                .child(div().text_sm().text_color(rgb(0xdce5f5)).child(label))
                .child(div().text_xs().text_color(rgb(0x7f8ba5)).child(time)),
        )
}
