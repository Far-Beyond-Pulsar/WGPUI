use wgpui::{
    ApplicationError, DispatchNodeId, EventResult, InputEvent, KeyBinding, NativeApplication, Rect,
    Styled, WindowOptions, actions, div, rgb,
};

actions!(demo, [Activate]);

fn main() -> Result<(), ApplicationError> {
    NativeApplication::new(WindowOptions::default(), |window| {
        let interaction = window.interaction();
        let button = interaction
            .hit_test()
            .insert(Rect::from_origin_size([0.0, 0.0], [160.0, 48.0]), 1);
        let node = DispatchNodeId(button.as_raw());
        interaction.dispatch_tree().on_input(node, |event| {
            if matches!(event, InputEvent::Click(_)) {
                EventResult::HANDLED
            } else {
                EventResult::IGNORED
            }
        });
        interaction.bind_key(KeyBinding::new("ctrl-enter", Activate, None));
        let _ = interaction.focus_manager();
        div().id("button").w(160.0).h(48.0).bg(rgb(0x2050a0))
    })
    .run()
}
