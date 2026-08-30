param(
    [switch]$Offline,
    [switch]$Locked,
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$manifest = Join-Path $PSScriptRoot "..\crates\wgpui-examples-2\Cargo.toml"
$exampleRoot = Join-Path $PSScriptRoot "..\crates\wgpui-examples-2\examples"
$logDirectory = if ($OutputDirectory) { $OutputDirectory } else { Join-Path $PSScriptRoot "..\target\examples-2-compile" }
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$cargoArgs = @("check", "--manifest-path", $manifest)
if ($Offline) { $cargoArgs += "--offline" }
if ($Locked) { $cargoArgs += "--locked" }

$entries = @(
    @{ Name = "native_elements"; Path = "native_elements.rs" }, @{ Name = "native_interaction"; Path = "native_interaction.rs" }, @{ Name = "karaoke_text"; Path = "karaoke_text.rs" }, @{ Name = "karaoke_app"; Path = "karaoke_app.rs" }, @{ Name = "karaoke_multiline"; Path = "karaoke_multiline.rs" }, @{ Name = "text_gradients"; Path = "text_gradients.rs" },
    @{ Name = "interactive_elements"; Path = "learn/interactive_elements.rs" }, @{ Name = "creating_components"; Path = "learn/creating_components.rs" }, @{ Name = "layout"; Path = "learn/layout.rs" }, @{ Name = "styling"; Path = "learn/styling.rs" }, @{ Name = "async_tasks"; Path = "learn/async_tasks.rs" }, @{ Name = "custom_drawing"; Path = "learn/custom_drawing.rs" }, @{ Name = "animation"; Path = "learn/animation.rs" }, @{ Name = "text"; Path = "learn/text.rs" }, @{ Name = "emoji_display"; Path = "learn/emoji_display.rs" }, @{ Name = "wgpu_surface"; Path = "learn/wgpu_surface.rs" }, @{ Name = "wgpu_surface_basic"; Path = "learn/wgpu_surface_basic.rs" }, @{ Name = "wgpu_surface_quad"; Path = "learn/wgpu_surface_quad.rs" }, @{ Name = "wgpu_surface_stress"; Path = "learn/wgpu_surface_stress.rs" }, @{ Name = "mouse_events"; Path = "learn/mouse_events.rs" }, @{ Name = "blur_showcase"; Path = "learn/blur_showcase.rs" }, @{ Name = "smooth_scrolling"; Path = "learn/smooth_scrolling.rs" }, @{ Name = "virtual_list"; Path = "learn/virtual_list.rs" },
    @{ Name = "data_table"; Path = "bench/data_table.rs" }, @{ Name = "plain_scroll_10k"; Path = "bench/plain_scroll_10k.rs" }, @{ Name = "paths_bench"; Path = "bench/paths_bench.rs" }, @{ Name = "pattern"; Path = "bench/pattern.rs" }, @{ Name = "shadow"; Path = "bench/shadow.rs" },
    @{ Name = "focus_visible"; Path = "legacy/focus_visible.rs" }, @{ Name = "gif_viewer"; Path = "legacy/gif_viewer.rs" }, @{ Name = "gradient"; Path = "legacy/gradient.rs" }, @{ Name = "hello_world"; Path = "legacy/hello_world.rs" }, @{ Name = "image_loading"; Path = "legacy/image_loading.rs" }, @{ Name = "input"; Path = "legacy/input.rs" }, @{ Name = "on_window_close_quit"; Path = "legacy/on_window_close_quit.rs" }, @{ Name = "opacity"; Path = "legacy/opacity.rs" }, @{ Name = "scrollable"; Path = "legacy/scrollable.rs" }, @{ Name = "svg"; Path = "legacy/svg/svg.rs" }, @{ Name = "tab_stop"; Path = "legacy/tab_stop.rs" }, @{ Name = "tree"; Path = "legacy/tree.rs" }, @{ Name = "uniform_list"; Path = "legacy/uniform_list.rs" }, @{ Name = "window"; Path = "legacy/window.rs" }, @{ Name = "window_positioning"; Path = "legacy/window_positioning.rs" }, @{ Name = "window_shadow"; Path = "legacy/window_shadow.rs" }, @{ Name = "image"; Path = "legacy/image/image.rs" }
)

$passed = 0
$failed = 0
foreach ($entry in $entries) {
    $output = & cargo @cargoArgs --example $entry.Name 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0) {
        $passed++
        Write-Output ("PASS|{0}|{1}|compile-only" -f $entry.Name, $entry.Path)
    } else {
        $failed++
        $category = if ($output -match "unresolved import|cannot find|no method named|trait bound") { "native-api-gap" } elseif ($output -match "linker|could not compile|failed to run custom build") { "toolchain-or-dependency" } else { "compile-error" }
        Write-Output ("FAIL|{0}|{1}|{2}" -f $entry.Name, $entry.Path, $category)
        $output | Set-Content (Join-Path $logDirectory ("{0}.compile.log" -f $entry.Name))
    }
}
Write-Output ("SUMMARY|total={0}|passed={1}|failed={2}|runtime-tested=0" -f $entries.Count, $passed, $failed)
if ($failed -gt 0) { exit 1 }
