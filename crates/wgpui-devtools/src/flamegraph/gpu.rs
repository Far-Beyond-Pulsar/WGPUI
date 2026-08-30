//! GPU timestamp adapter boundary; timestamp ownership stays in `wgpui-wgpu`.
use wgpui_core::hooks::InstrumentationHooks;
pub fn record_timestamp(
    hooks: &dyn InstrumentationHooks,
    name: &'static str,
    start: u64,
    end: u64,
) {
    hooks.gpu_timestamp(name, start, end);
}
