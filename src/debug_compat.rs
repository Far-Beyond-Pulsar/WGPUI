#![allow(missing_docs)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
static LAYER_DEBUG: AtomicBool = AtomicBool::new(false);
static HUD: AtomicBool = AtomicBool::new(false);
static SLOW: AtomicBool = AtomicBool::new(false);
static FPS: AtomicBool = AtomicBool::new(false);
static FORCE: AtomicBool = AtomicBool::new(false);
static OCCLUSION: AtomicBool = AtomicBool::new(false);
static OCCLUSION_VIS: AtomicBool = AtomicBool::new(false);
static RASTERIZE_ABOVE: AtomicUsize = AtomicUsize::new(0);
static EVICT_AFTER: AtomicUsize = AtomicUsize::new(0);
pub fn is_layer_debug_enabled()->bool { LAYER_DEBUG.load(Ordering::Relaxed) }
pub fn set_layer_debug_enabled(v:bool){LAYER_DEBUG.store(v,Ordering::Relaxed)}
pub fn is_hud_enabled()->bool { HUD.load(Ordering::Relaxed) }
pub fn set_hud_enabled(v:bool){HUD.store(v,Ordering::Relaxed)}
pub fn is_slow_frame_flash_enabled()->bool { SLOW.load(Ordering::Relaxed) }
pub fn set_slow_frame_flash_enabled(v:bool){SLOW.store(v,Ordering::Relaxed)}
pub fn is_layer_fps_enabled()->bool { FPS.load(Ordering::Relaxed) }
pub fn set_layer_fps_enabled(v:bool){FPS.store(v,Ordering::Relaxed)}
pub fn is_force_rasterize_all()->bool { FORCE.load(Ordering::Relaxed) }
pub fn set_force_rasterize_all(v:bool){FORCE.store(v,Ordering::Relaxed)}
pub fn rasterize_above_override()->Option<usize> { match RASTERIZE_ABOVE.load(Ordering::Relaxed) { 0 => None, v => Some(v) } }
pub fn set_rasterize_above_override(v:Option<usize>) { RASTERIZE_ABOVE.store(v.unwrap_or(0), Ordering::Relaxed); }
pub fn evict_after_frames_override()->Option<u32> { match EVICT_AFTER.load(Ordering::Relaxed) { 0 => None, v => Some(v as u32) } }
pub fn set_evict_after_frames_override(v:Option<u32>) { EVICT_AFTER.store(v.unwrap_or(0) as usize, Ordering::Relaxed); }
pub fn is_occlusion_disabled()->bool { OCCLUSION.load(Ordering::Relaxed) }
pub fn set_occlusion_disabled(v:bool){OCCLUSION.store(v,Ordering::Relaxed)}
pub fn is_occlusion_visualizer_enabled()->bool { OCCLUSION_VIS.load(Ordering::Relaxed) }
pub fn set_occlusion_visualizer_enabled(v:bool){OCCLUSION_VIS.store(v,Ordering::Relaxed)}
