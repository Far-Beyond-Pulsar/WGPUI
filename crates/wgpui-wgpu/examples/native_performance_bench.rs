//! Reproducible native retained-pipeline baseline.
//!
//! The benchmark opens a real native surface, warms each workload, presents
//! timed frames, and writes one JSON document to stdout. It measures the
//! frontend stages through `FrameLoop::draw_profiled`, uses the renderer's
//! existing upload/compute counters, and times the real `SurfaceTexture`
//! present call.
//!
//! ```text
//! cargo run -p wgpui-wgpu --release --features devtools --example native_performance_bench -- --diagnostics both
//! ```

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use wgpui_core::boundary::policy::{BoundaryPolicy, Buffering};
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::invalidation::request::FrameSignals;
use wgpui_core::patch::emit::{Emission, Emit, EmitContext};
use wgpui_core::patch::primitive::{Material, Quad, Shadow};
use wgpui_core::reconcile::description::{Description, ElementId};
use wgpui_core::reconcile::diff_key::{ReconcileKey, compare_by_equality};
use wgpui_core::scene::layer::LayerId;
use wgpui_layout::taffy_tree::{Dimension, Display, FlexDirection, LayoutSize, LayoutStyle};
use wgpui_wgpu::render::device::ComputeContext;
use wgpui_wgpu::render::draw::DrawMode;
use wgpui_wgpu::render::frame::RenderTarget;
use wgpui_wgpu::window::frame_loop::{FrameLoop, LoopFrame, LoopInput};
use wgpui_wgpu::window::{Acquired, WindowSurface};

const DEFAULT_WIDTH: u32 = 960;
const DEFAULT_HEIGHT: u32 = 640;
const DEFAULT_SIBLINGS: usize = 32;
const DEFAULT_DEPTH: usize = 4;
const DEFAULT_FRAMES: usize = 24;
const DEFAULT_WARMUP: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    Steady,
    Scroll,
    Continuous,
}

impl Scenario {
    const ALL: [Scenario; 3] = [Self::Steady, Self::Scroll, Self::Continuous];

    const fn name(self) -> &'static str {
        match self {
            Self::Steady => "steady",
            Self::Scroll => "scroll",
            Self::Continuous => "continuous",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiagnosticSetting {
    Off,
    On,
}

impl DiagnosticSetting {
    const fn enabled(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Clone, Debug)]
struct Options {
    siblings: usize,
    depth: usize,
    frames: usize,
    warmup: usize,
    diagnostics: String,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut options = Self {
            siblings: DEFAULT_SIBLINGS,
            depth: DEFAULT_DEPTH,
            frames: DEFAULT_FRAMES,
            warmup: DEFAULT_WARMUP,
            diagnostics: "both".to_string(),
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--siblings" => {
                    options.siblings = next_argument(&mut arguments, "--siblings")?
                        .parse()
                        .map_err(|error| format!("invalid siblings: {error}"))?;
                }
                "--depth" => {
                    options.depth = next_argument(&mut arguments, "--depth")?
                        .parse()
                        .map_err(|error| format!("invalid depth: {error}"))?;
                }
                "--frames" => {
                    options.frames = next_argument(&mut arguments, "--frames")?
                        .parse()
                        .map_err(|error| format!("invalid frames: {error}"))?;
                }
                "--warmup" => {
                    options.warmup = next_argument(&mut arguments, "--warmup")?
                        .parse()
                        .map_err(|error| format!("invalid warmup: {error}"))?;
                }
                "--diagnostics" => {
                    options.diagnostics = next_argument(&mut arguments, "--diagnostics")?
                }
                other => return Err(format!("unrecognised argument {other:?}")),
            }
        }
        if options.siblings == 0 || options.frames == 0 {
            return Err("siblings and frames must be greater than zero".to_string());
        }
        if options.warmup.checked_add(options.frames).is_none() {
            return Err("warmup plus frames is too large".to_string());
        }
        if !matches!(options.diagnostics.as_str(), "off" | "on" | "both") {
            return Err("--diagnostics must be off, on, or both".to_string());
        }
        Ok(options)
    }

    fn settings(&self) -> Vec<DiagnosticSetting> {
        let settings = match self.diagnostics.as_str() {
            "off" => vec![DiagnosticSetting::Off],
            "on" => vec![DiagnosticSetting::On],
            _ => vec![DiagnosticSetting::Off, DiagnosticSetting::On],
        };
        #[cfg(not(feature = "devtools"))]
        if settings.iter().any(|setting| setting.enabled()) {
            return vec![DiagnosticSetting::Off];
        }
        settings
    }
}

fn next_argument(
    arguments: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{name} needs a value"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BenchmarkKey {
    revision: u64,
}

impl ReconcileKey for BenchmarkKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        compare_by_equality(self, previous, Invalidation::DISPLAY)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct BenchmarkPaint {
    color: [f32; 4],
}

impl Emit for BenchmarkPaint {
    fn emit(&self, context: &EmitContext, emission: &mut Emission) {
        emission.shadow(Shadow {
            origin: [context.bounds.x, context.bounds.y],
            size: [context.bounds.width, context.bounds.height],
            color: [0.0, 0.0, 0.0, 0.28],
            corner_radii: [6.0; 4],
            blur_radius: 4.0,
        });
        emission.quad(Quad {
            origin: [context.bounds.x, context.bounds.y],
            size: [context.bounds.width, context.bounds.height],
            background: self.color,
            border_color: [0.85, 0.9, 1.0, 1.0],
            corner_radii: [6.0; 4],
            border_widths: [1.0; 4],
            material: Material::Solid,
        });
    }
}

struct BenchmarkState {
    frame_loop: FrameLoop,
    scroll_layer: Option<LayerId>,
    mode: DrawMode,
}

impl BenchmarkState {
    fn new(context: &ComputeContext) -> Self {
        Self {
            frame_loop: FrameLoop::new(&context.device),
            scroll_layer: None,
            mode: DrawMode::best_available(context.indirect),
        }
    }
}

fn percent_style() -> LayoutStyle {
    LayoutStyle {
        size: LayoutSize {
            width: Dimension::percent(1.0),
            height: Dimension::percent(1.0),
        },
        ..LayoutStyle::default()
    }
}

fn row_style() -> LayoutStyle {
    LayoutStyle {
        display: Display::Flex,
        flex_direction: FlexDirection::Column,
        size: LayoutSize {
            width: Dimension::percent(1.0),
            height: Dimension::length(52.0),
        },
        ..LayoutStyle::default()
    }
}

fn nested_description(sibling: usize, level: usize, depth: usize, revision: u64) -> Description {
    if level < depth {
        return Description::new::<BenchmarkGroup>()
            .id(ElementId::Integer((sibling as u64) << 32 | level as u64))
            .diff_key(BenchmarkKey { revision: 0 })
            .style(row_style())
            .child(nested_description(sibling, level + 1, depth, revision));
    }

    let red = 0.25 + (sibling % 7) as f32 * 0.06;
    let green = 0.35 + (sibling % 5) as f32 * 0.07;
    let blue = if sibling == 0 && revision != 0 {
        0.45 + (revision % 12) as f32 * 0.025
    } else {
        0.65
    };
    let paint_revision = if sibling == 0 { revision } else { 0 };
    let paint = Description::new::<BenchmarkSurface>()
        .id(ElementId::Integer(0x8000_0000 | sibling as u64))
        .diff_key(BenchmarkKey {
            revision: paint_revision,
        })
        .style(row_style())
        .emit(BenchmarkPaint {
            color: [red, green, blue, 1.0],
        });
    let text = Description::raw_text(format!("surface {sibling:03} / depth {depth}"))
        .text_metrics(Some(14.0), Some([0.96, 0.97, 1.0, 1.0]))
        .style(LayoutStyle {
            size: LayoutSize {
                width: Dimension::percent(1.0),
                height: Dimension::length(24.0),
            },
            ..LayoutStyle::default()
        });
    Description::new::<BenchmarkLeaf>()
        .id(ElementId::Integer(0x4000_0000 | sibling as u64))
        .diff_key(BenchmarkKey { revision: 0 })
        .style(row_style())
        .child(paint)
        .child(text)
}

fn build_description(options: &Options, scenario: Scenario, frame: usize) -> Description {
    let scroll_offset = if scenario == Scenario::Scroll {
        [0.0, -(frame as f32 * 2.0)]
    } else {
        [0.0, 0.0]
    };
    let scroll = Description::new::<BenchmarkScroll>()
        .id("scroll-root")
        .diff_key(BenchmarkKey { revision: 0 })
        .boundary_with_policy(BoundaryPolicy {
            rasterize_above: usize::MAX,
            buffering: Buffering::Margin(None),
            ..BoundaryPolicy::default()
        })
        .scroll_offset(scroll_offset)
        .clip_children()
        .style(percent_style())
        .children((0..options.siblings).map(|sibling| {
            let revision = if scenario == Scenario::Continuous {
                frame as u64 + 1
            } else {
                0
            };
            nested_description(sibling, 0, options.depth, revision)
        }));
    Description::new::<BenchmarkRoot>()
        .id("benchmark-root")
        .diff_key(BenchmarkKey { revision: 0 })
        .style(percent_style())
        .child(scroll)
}

#[derive(Clone, Copy, Debug, Default)]
struct StageSample {
    description_build: Duration,
    reconciliation: Duration,
    layout: Duration,
    shared_walk: Duration,
    emission: Duration,
    damage: Duration,
    uploads: Duration,
    visibility: Duration,
    present: Duration,
}

impl StageSample {
    fn entries(self) -> [(&'static str, Duration); 9] {
        [
            ("description_build", self.description_build),
            ("reconciliation", self.reconciliation),
            ("layout", self.layout),
            ("shared_walk", self.shared_walk),
            ("emission", self.emission),
            ("damage", self.damage),
            ("uploads", self.uploads),
            ("visibility", self.visibility),
            ("present", self.present),
        ]
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct AccumulatedCounters {
    frames: u64,
    idle_frames: u64,
    presented: u64,
    nodes_visited: u64,
    nodes_reused: u64,
    nodes_rebuilt: u64,
    nodes_emitted: u64,
    nodes_skipped: u64,
    patch_operations: u64,
    dirty_layers: u64,
    upload_calls: u64,
    upload_bytes: u64,
    layers_recomputed: u64,
    primitives_resident: u64,
    draw_calls: u64,
}

impl AccumulatedCounters {
    fn add(&mut self, frame: &LoopFrame) {
        self.frames += 1;
        self.idle_frames += u64::from(frame.was_idle());
        self.presented += 1;
        self.nodes_visited += frame.reconciled.visited as u64;
        self.nodes_reused += frame.reconciled.reused as u64;
        self.nodes_rebuilt += frame.reconciled.rebuilt as u64;
        self.nodes_emitted += frame.emission.stats.nodes_emitted as u64;
        self.nodes_skipped += frame.emission.stats.nodes_skipped as u64;
        self.patch_operations += frame.emission.patch.len() as u64;
        self.dirty_layers += frame.dirty_layers.len() as u64;
        self.upload_calls += frame.frame.scene_upload_calls as u64;
        self.upload_bytes += frame.frame.scene_upload_bytes;
        self.layers_recomputed += frame.frame.layers_recomputed as u64;
        self.primitives_resident += frame.frame.primitives_resident as u64;
        self.draw_calls += frame.frame.stats.draw_calls_issued as u64;
    }
}

#[derive(Serialize)]
struct StageStats {
    samples: usize,
    mean_ns: u64,
    median_ns: u64,
    best_ns: u64,
    total_ns: u64,
}

#[derive(Serialize)]
struct CostCenter {
    stage: String,
    total_ns: u64,
    share_percent: f64,
}

#[derive(Serialize)]
struct DiagnosticTimer {
    count: u64,
    total_ns: u64,
    max_ns: u64,
}

#[derive(Serialize)]
struct RunReport {
    scenario: &'static str,
    diagnostics: bool,
    frames: usize,
    warmup: usize,
    stages: BTreeMap<String, StageStats>,
    cost_centers: Vec<CostCenter>,
    counters: AccumulatedCounters,
    diagnostic_timers: BTreeMap<String, DiagnosticTimer>,
    diagnostic_counters: BTreeMap<String, u64>,
}

#[derive(Serialize)]
struct BenchmarkReport {
    schema: u32,
    adapter: String,
    software_adapter: bool,
    viewport: [u32; 2],
    siblings: usize,
    depth: usize,
    draw_mode: String,
    runs: Vec<RunReport>,
    limitations: [&'static str; 4],
}

struct ActiveRun {
    scenario: Scenario,
    diagnostics: DiagnosticSetting,
    state: BenchmarkState,
    frame: usize,
    samples: Vec<StageSample>,
    counters: AccumulatedCounters,
}

impl ActiveRun {
    fn new(context: &ComputeContext, scenario: Scenario, diagnostics: DiagnosticSetting) -> Self {
        Self {
            scenario,
            diagnostics,
            state: BenchmarkState::new(context),
            frame: 0,
            samples: Vec::new(),
            counters: AccumulatedCounters::default(),
        }
    }
}

struct Live {
    surface: WindowSurface,
    context: ComputeContext,
    options: Options,
    settings: Vec<DiagnosticSetting>,
    scenarios: Vec<Scenario>,
    next_scenario: usize,
    active: ActiveRun,
}

struct BenchmarkApp {
    options: Options,
    reports: Vec<RunReport>,
    live: Option<Live>,
    failure: Option<String>,
    adapter: Option<String>,
    software_adapter: bool,
    draw_mode: String,
}

impl BenchmarkApp {
    fn start_run(live: &mut Live, setting: DiagnosticSetting, scenario: Scenario) {
        configure_diagnostics(setting);
        live.active = ActiveRun::new(&live.context, scenario, setting);
    }

    fn finish_run(&mut self, live: &mut Live) {
        let active = &live.active;
        self.reports.push(make_report(active, &live.options));
        live.next_scenario += 1;
        if live.next_scenario >= live.scenarios.len() {
            live.settings.remove(0);
            live.next_scenario = 0;
        }
        if let Some(setting) = live.settings.first().copied()
            && let Some(scenario) = live.scenarios.get(live.next_scenario).copied()
        {
            Self::start_run(live, setting, scenario);
        }
    }

    fn draw(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let Some(mut live) = self.live.take() else {
            return;
        };
        let texture = match live.surface.acquire(&live.context.device) {
            Acquired::Frame(texture) => texture,
            Acquired::Skipped(_) => {
                live.surface.window().request_redraw();
                self.live = Some(live);
                return;
            }
            Acquired::Lost => {
                self.failure = Some("surface acquire failed".to_string());
                event_loop.exit();
                return;
            }
        };
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (width, height) = live.surface.size();
        #[cfg(feature = "devtools")]
        if live.active.frame == live.options.warmup {
            wgpui_devtools::render_stats::reset();
        }
        let build_started = Instant::now();
        let description = build_description(&live.options, live.active.scenario, live.active.frame);
        let description_build = build_started.elapsed();
        let mut signals = FrameSignals::new();
        if live.active.scenario == Scenario::Scroll
            && let Some(layer) = live.active.state.scroll_layer
        {
            signals.scrolled(layer);
        }
        let target = RenderTarget {
            view: &view,
            width,
            height,
            clear: wgpu::Color::BLACK,
            source: Some(&texture.texture),
        };
        let result = live.active.state.frame_loop.draw_profiled(
            &live.context.device,
            &live.context.queue,
            description,
            &LoopInput {
                atlas: None,
                target: &target,
                mode: live.active.state.mode,
                signals: &signals,
                composites: &[],
            },
        );
        let (frame, timing) = match result {
            Ok(frame) => frame,
            Err(error) => {
                self.failure = Some(format!("frame failed: {error}"));
                event_loop.exit();
                return;
            }
        };
        if live.active.state.scroll_layer.is_none() {
            live.active.state.scroll_layer = frame
                .emission
                .composites
                .iter()
                .find(|composite| composite.boundary != wgpui_core::scene::layer::BoundaryId::ROOT)
                .map(|composite| composite.layer);
        }
        let present_started = Instant::now();
        live.surface.present(&live.context.queue, texture);
        let present = present_started.elapsed();

        let sample = StageSample {
            description_build: description_build + timing.description_build,
            reconciliation: timing.reconciliation,
            layout: timing.layout,
            shared_walk: timing.shared_walk,
            emission: timing.emission,
            damage: timing.damage,
            uploads: frame.frame.timing.upload,
            visibility: frame.frame.timing.compute,
            present,
        };
        live.active.frame += 1;
        if live.active.frame > live.options.warmup {
            #[cfg(feature = "devtools")]
            if live.active.diagnostics.enabled() {
                record_diagnostic_sample(sample);
            }
            live.active.samples.push(sample);
            live.active.counters.add(&frame);
        }
        if live.active.frame >= live.options.warmup + live.options.frames {
            self.finish_run(&mut live);
            if live.settings.is_empty() {
                event_loop.exit();
                return;
            }
        }
        live.surface.window().request_redraw();
        self.live = Some(live);
    }
}

impl winit::application::ApplicationHandler for BenchmarkApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.live.is_some() {
            return;
        }
        let attributes = winit::window::Window::default_attributes()
            .with_title("WGPUI native performance baseline")
            .with_inner_size(winit::dpi::PhysicalSize::new(DEFAULT_WIDTH, DEFAULT_HEIGHT));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.failure = Some(format!("create window failed: {error}"));
                event_loop.exit();
                return;
            }
        };
        let (surface, context) = match WindowSurface::new(Arc::clone(&window)) {
            Ok(pair) => pair,
            Err(error) => {
                self.failure = Some(format!("create surface failed: {error}"));
                event_loop.exit();
                return;
            }
        };
        let settings = self.options.settings();
        let scenarios = Scenario::ALL.to_vec();
        let setting = settings.first().copied().unwrap_or(DiagnosticSetting::Off);
        let scenario = scenarios.first().copied().unwrap_or(Scenario::Steady);
        let active = ActiveRun::new(&context, scenario, setting);
        let mut live = Live {
            surface,
            context,
            options: self.options.clone(),
            settings,
            scenarios,
            next_scenario: 0,
            active,
        };
        self.adapter = Some(live.context.describe());
        self.software_adapter = live.context.is_software();
        self.draw_mode = live.active.state.mode.name().to_string();
        Self::start_run(&mut live, setting, scenario);
        self.live = Some(live);
        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::CloseRequested => event_loop.exit(),
            winit::event::WindowEvent::RedrawRequested => self.draw(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(live) = self.live.as_ref() {
            live.surface.window().request_redraw();
        }
    }
}

#[cfg(feature = "devtools")]
fn configure_diagnostics(setting: DiagnosticSetting) {
    wgpui_devtools::render_stats::set_enabled(setting.enabled());
    wgpui_devtools::render_stats::reset();
}

#[cfg(not(feature = "devtools"))]
fn configure_diagnostics(_setting: DiagnosticSetting) {}

#[cfg(feature = "devtools")]
fn record_diagnostic_sample(sample: StageSample) {
    for (name, duration) in sample.entries() {
        wgpui_devtools::render_stats::record(name, duration);
    }
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn stage_stats(samples: &[StageSample]) -> BTreeMap<String, StageStats> {
    let mut durations: BTreeMap<&'static str, Vec<Duration>> = BTreeMap::new();
    for sample in samples.iter().copied() {
        for (name, duration) in sample.entries() {
            durations.entry(name).or_default().push(duration);
        }
    }
    durations
        .into_iter()
        .map(|(name, mut values)| {
            values.sort_unstable();
            let total_ns = values
                .iter()
                .copied()
                .map(duration_ns)
                .fold(0, u64::saturating_add);
            let best_ns = values.first().copied().map(duration_ns).unwrap_or(0);
            let median_ns = values
                .get(values.len() / 2)
                .copied()
                .map(duration_ns)
                .unwrap_or(0);
            let mean_ns = if values.is_empty() {
                0
            } else {
                total_ns / values.len() as u64
            };
            (
                name.to_string(),
                StageStats {
                    samples: values.len(),
                    mean_ns,
                    median_ns,
                    best_ns,
                    total_ns,
                },
            )
        })
        .collect()
}

fn make_report(active: &ActiveRun, options: &Options) -> RunReport {
    let stages = stage_stats(&active.samples);
    let total_ns = stages.values().map(|stats| stats.total_ns).sum::<u64>();
    let mut cost_centers: Vec<_> = stages
        .iter()
        .map(|(stage, stats)| CostCenter {
            stage: stage.clone(),
            total_ns: stats.total_ns,
            share_percent: if total_ns == 0 {
                0.0
            } else {
                stats.total_ns as f64 * 100.0 / total_ns as f64
            },
        })
        .collect();
    cost_centers.sort_by_key(|center| std::cmp::Reverse(center.total_ns));
    let (diagnostic_timers, diagnostic_counters) = diagnostic_snapshot();
    RunReport {
        scenario: active.scenario.name(),
        diagnostics: active.diagnostics.enabled(),
        frames: options.frames,
        warmup: options.warmup,
        stages,
        cost_centers,
        counters: active.counters.clone(),
        diagnostic_timers,
        diagnostic_counters,
    }
}

#[cfg(feature = "devtools")]
fn diagnostic_snapshot() -> (BTreeMap<String, DiagnosticTimer>, BTreeMap<String, u64>) {
    let snapshot = wgpui_devtools::render_stats::snapshot();
    let timers = snapshot
        .timers
        .into_iter()
        .map(|(name, timer)| {
            (
                name.to_string(),
                DiagnosticTimer {
                    count: timer.count,
                    total_ns: duration_ns(timer.total),
                    max_ns: duration_ns(timer.max),
                },
            )
        })
        .collect();
    (
        timers,
        snapshot
            .counters
            .into_iter()
            .map(|(name, count)| (name.to_string(), count))
            .collect(),
    )
}

#[cfg(not(feature = "devtools"))]
fn diagnostic_snapshot() -> (BTreeMap<String, DiagnosticTimer>, BTreeMap<String, u64>) {
    (BTreeMap::new(), BTreeMap::new())
}

fn main() {
    let options = match Options::parse() {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}");
            return;
        }
    };
    #[cfg(not(feature = "devtools"))]
    if options.diagnostics != "off" {
        eprintln!(
            "diagnostics on requested, but this example was built without --features devtools; running diagnostics off only"
        );
    }
    let event_loop = match winit::event_loop::EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(error) => {
            eprintln!("create event loop failed: {error}");
            return;
        }
    };
    let mut app = BenchmarkApp {
        options,
        reports: Vec::new(),
        live: None,
        failure: None,
        adapter: None,
        software_adapter: false,
        draw_mode: String::new(),
    };
    if let Err(error) = event_loop.run_app(&mut app) {
        eprintln!("benchmark event loop failed: {error}");
        return;
    }
    if let Some(error) = app.failure {
        eprintln!("benchmark failed: {error}");
        return;
    }
    if app.reports.is_empty() {
        eprintln!("benchmark produced no reports");
        return;
    }
    let report = BenchmarkReport {
        schema: 1,
        adapter: app
            .adapter
            .unwrap_or_else(|| "adapter unavailable".to_string()),
        software_adapter: app.software_adapter,
        viewport: [DEFAULT_WIDTH, DEFAULT_HEIGHT],
        siblings: app.options.siblings,
        depth: app.options.depth,
        draw_mode: app.draw_mode,
        runs: app.reports,
        limitations: [
            "The visibility field is FrameRenderer's existing dirty-layer ordering/occlusion CPU dispatch timing; tiled visibility remains covered by phase45_tiling_bench.",
            "description_build combines the benchmark's Description construction with FrameLoop raw-text materialization; it does not include window event dispatch.",
            "uploads reports FrameRenderer's scene-arena upload timing; glyph-atlas synchronization has no separate existing timing hook.",
            "The benchmark uses a real native surface and present call; present timing includes the configured surface present-mode pacing.",
        ],
    };
    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(error) => eprintln!("serialize benchmark report failed: {error}"),
    }
}

struct BenchmarkRoot;
struct BenchmarkScroll;
struct BenchmarkGroup;
struct BenchmarkLeaf;
struct BenchmarkSurface;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_report_keeps_all_requested_measurements() -> Result<(), String> {
        let sample = StageSample {
            description_build: Duration::from_nanos(1),
            reconciliation: Duration::from_nanos(2),
            layout: Duration::from_nanos(3),
            shared_walk: Duration::from_nanos(4),
            emission: Duration::from_nanos(5),
            damage: Duration::from_nanos(6),
            uploads: Duration::from_nanos(7),
            visibility: Duration::from_nanos(8),
            present: Duration::from_nanos(9),
        };
        let report = stage_stats(&[sample]);
        let expected_stages = [
            "description_build",
            "reconciliation",
            "layout",
            "shared_walk",
            "emission",
            "damage",
            "uploads",
            "visibility",
            "present",
        ];

        assert_eq!(report.len(), expected_stages.len());
        for (index, stage) in expected_stages.iter().enumerate() {
            let stats = report
                .get(*stage)
                .ok_or_else(|| format!("missing stage {stage}"))?;
            assert_eq!(stats.samples, 1);
            assert_eq!(stats.mean_ns, (index + 1) as u64);
            assert_eq!(stats.median_ns, (index + 1) as u64);
            assert_eq!(stats.best_ns, (index + 1) as u64);
        }
        Ok(())
    }

    #[test]
    fn workload_has_siblings_nesting_text_and_continuous_invalidation() -> Result<(), String> {
        let options = Options {
            siblings: 3,
            depth: 2,
            frames: 1,
            warmup: 0,
            diagnostics: "off".to_string(),
        };
        let steady = build_description(&options, Scenario::Steady, 0);
        let continuous = build_description(&options, Scenario::Continuous, 1);

        let steady_scroll = steady
            .child_descriptions()
            .first()
            .ok_or("root should contain the scroll boundary")?;
        assert!(steady_scroll.is_boundary());
        assert_eq!(steady_scroll.child_descriptions().len(), options.siblings);
        assert_eq!(steady_scroll.scroll_offset_of(), [0.0, 0.0]);

        let nested = steady_scroll
            .child_descriptions()
            .first()
            .ok_or("scroll boundary should contain a sibling")?;
        assert_eq!(nested.child_descriptions().len(), 1);
        let nested_group = nested
            .child_descriptions()
            .first()
            .ok_or("nested group should contain another group")?;
        let leaf = nested_group
            .child_descriptions()
            .first()
            .ok_or("leaf should be below the requested depth")?;
        assert_eq!(leaf.child_descriptions().len(), 2);

        let steady_paint = leaf
            .child_descriptions()
            .first()
            .ok_or("leaf should contain a paint description")?;
        let steady_text = leaf
            .child_descriptions()
            .get(1)
            .ok_or("leaf should contain text")?;
        assert_eq!(steady_paint.child_descriptions().len(), 0);
        assert_eq!(steady_text.child_descriptions().len(), 0);

        let continuous_scroll = continuous
            .child_descriptions()
            .first()
            .ok_or("continuous root should contain the scroll boundary")?;
        assert_eq!(continuous_scroll.scroll_offset_of(), [0.0, 0.0]);
        assert_eq!(
            BenchmarkKey { revision: 0 }.compare(&BenchmarkKey { revision: 0 }),
            Invalidation::empty()
        );
        assert_eq!(
            BenchmarkKey { revision: 1 }.compare(&BenchmarkKey { revision: 0 }),
            Invalidation::DISPLAY
        );
        Ok(())
    }
}
