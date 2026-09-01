<#
.SYNOPSIS
    Launches all WGPUI examples concurrently and aggregates runtime diagnostics.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\script\run-examples-2.ps1 -Locked

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\script\run-examples-2.ps1 -NoBuild -Examples uniform_list,window

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\script\run-examples-2.ps1 -NativeDebug -NativeDebuggerPath C:\path\to\cdb.exe
#>
param(
    [switch]$NoBuild,
    [switch]$Debug,
    [switch]$Locked,
    [switch]$NativeDebug,
    [string]$NativeDebuggerPath = "",
    [string]$OutputDirectory = "",
    [string[]]$Examples = @()
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$profileName = if ($Debug) { "debug" } else { "release" }
$binaryDirectory = Join-Path $repositoryRoot ("target\{0}\examples" -f $profileName)
$defaultOutputDirectory = Join-Path $repositoryRoot ("target\examples-2-run\{0}" -f (Get-Date -Format "yyyyMMdd-HHmmss-fff"))
$runDirectory = if ($OutputDirectory) {
    if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
        [System.IO.Path]::GetFullPath($OutputDirectory)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDirectory))
    }
} else {
    $defaultOutputDirectory
}
$null = New-Item -ItemType Directory -Force -Path $runDirectory

$allExamples = @(
    "native_elements", "native_interaction", "karaoke_text", "karaoke_app", "karaoke_multiline", "text_gradients",
    "interactive_elements", "creating_components", "layout", "styling", "async_tasks", "custom_drawing", "animation", "text",
    "emoji_display", "wgpu_surface", "wgpu_surface_basic", "wgpu_surface_quad", "wgpu_surface_stress", "mouse_events", "blur_showcase",
    "smooth_scrolling", "virtual_list", "data_table", "plain_scroll_10k", "paths_bench", "pattern", "shadow",
    "focus_visible", "gif_viewer", "gradient", "hello_world", "image_loading", "input", "on_window_close_quit", "opacity",
    "scrollable", "svg", "tab_stop", "tree", "uniform_list", "window", "window_positioning", "window_shadow", "image"
)

$selectedExamples = if ($Examples.Count -eq 0) { $allExamples } else { $Examples }
$unknownExamples = @($selectedExamples | Where-Object { $_ -notin $allExamples })
if ($unknownExamples.Count -gt 0) {
    throw "Unknown example(s): $($unknownExamples -join ', ')"
}

function Resolve-NativeDebugger {
    if ($NativeDebuggerPath) {
        $resolvedPath = (Resolve-Path -LiteralPath $NativeDebuggerPath -ErrorAction Stop).Path
        if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
            throw "Native debugger is not a file: $resolvedPath"
        }
        return $resolvedPath
    }

    $command = Get-Command cdb.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Path
    }

    $knownPaths = @(
        (Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\Debuggers\x64\cdb.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\Debuggers\x86\cdb.exe"),
        (Join-Path $env:ProgramFiles "Windows Kits\10\Debuggers\x64\cdb.exe"),
        (Join-Path $env:ProgramFiles "Windows Kits\10\Debuggers\x86\cdb.exe")
    )
    foreach ($knownPath in $knownPaths) {
        if ($knownPath -and (Test-Path -LiteralPath $knownPath -PathType Leaf)) {
            return $knownPath
        }
    }

    throw "-NativeDebug requires cdb.exe. Install Windows Debugging Tools or pass -NativeDebuggerPath <path-to-cdb.exe>."
}

function Get-DiagnosticDetails {
    param(
        [string]$StandardOutput,
        [string]$StandardError,
        [string]$NativeOutput
    )

    $allOutputLines = @($StandardOutput -split "`r?`n") + @($StandardError -split "`r?`n")
    $rustLines = @($allOutputLines | Where-Object { $_ -match "panic|panicked at|stack backtrace:|^\s+\d+:|^\s+at " })
    $nativeLines = @($NativeOutput -split "`r?`n" | Where-Object { $_ -match "Exception|FAULTING_IP|Access violation|\.ecxr|^\s*[0-9a-f`?]+\s+" })
    [ordered]@{
        panic = @($allOutputLines | Where-Object { $_ -match "panicked at|panic!|thread '.*' panicked|RefCell already borrowed|Validation Error|wgpu error|Caused by:|required but not enabled" } | Select-Object -First 8)
        errors = @($allOutputLines | Where-Object { $_ -match "RefCell already borrowed|Validation Error|wgpu error|Caused by:|required but not enabled" } | Select-Object -First 8)
        rust_stack = @($rustLines | Where-Object { $_ -match "stack backtrace:|^\s+\d+:|^\s+at " } | Select-Object -First 24)
        native_exception = @($nativeLines | Select-Object -First 12)
    }
}

function Invoke-Build {
    if ($NoBuild) {
        return
    }

    $cargoArguments = @("build", "--profile", $profileName, "-p", "wgpui-examples-2", "--examples", "-j", "50")
    if ($Locked) {
        $cargoArguments += "--locked"
    }

    Push-Location $repositoryRoot
    try {
        & cargo @cargoArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Cargo build failed with exit code $LASTEXITCODE."
        }
    } finally {
        Pop-Location
    }
}

Invoke-Build

$debuggerPath = if ($NativeDebug) { Resolve-NativeDebugger } else { $null }
$debugCommand = ".symfix; .reload; sxe av; sxe eh; g; .ecxr; k; q"
$launches = [System.Collections.Generic.List[object]]::new()

foreach ($exampleName in $selectedExamples) {
    $binaryPath = Join-Path $binaryDirectory ("{0}.exe" -f $exampleName)
    if (-not (Test-Path -LiteralPath $binaryPath -PathType Leaf)) {
        throw "Missing binary for ${exampleName}: $binaryPath. Remove -NoBuild or build the selected profile first."
    }

    $stdoutPath = Join-Path $runDirectory ("{0}.stdout.log" -f $exampleName)
    $stderrPath = Join-Path $runDirectory ("{0}.stderr.log" -f $exampleName)
    $nativeLogPath = if ($NativeDebug) {
        Join-Path $runDirectory ("{0}.native.log" -f $exampleName)
    } else {
        $null
    }

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WorkingDirectory = $repositoryRoot
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.Environment["RUST_BACKTRACE"] = "full"
    $startInfo.Environment["RUST_LIB_BACKTRACE"] = "1"

    if ($NativeDebug) {
        $startInfo.FileName = $debuggerPath
        foreach ($argument in @("-o", "-g", "-G", ("-logo:{0}" -f $nativeLogPath), "-c", $debugCommand, $binaryPath)) {
            $null = $startInfo.ArgumentList.Add($argument)
        }
    } else {
        $startInfo.FileName = $binaryPath
    }

    $record = [ordered]@{
        name = $exampleName
        binary = $binaryPath
        pid = $null
        debugger_pid = $null
        started_at = (Get-Date).ToString("o")
        ended_at = $null
        exit_code = $null
        status = "launching"
        rust_backtrace = $false
        native_exception = $false
        stdout = $stdoutPath
        stderr = $stderrPath
        native_log = $nativeLogPath
        launch_error = $null
        diagnostics = $null
        debugger_exit_code = $null
    }

    try {
        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $startInfo
        if (-not $process.Start()) {
            throw "Process.Start returned false."
        }
        $record.pid = $process.Id
        if ($NativeDebug) {
            $record.debugger_pid = $process.Id
        }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $launches.Add([pscustomobject]@{
            Record = [pscustomobject]$record
            Process = $process
            StdoutTask = $stdoutTask
            StderrTask = $stderrTask
        })
        Write-Output ("LAUNCHED|{0}|pid={1}" -f $exampleName, $process.Id)
    } catch {
        $record.status = "launch-failed"
        $record.launch_error = $_.Exception.Message
        $record.ended_at = (Get-Date).ToString("o")
        $launches.Add([pscustomobject]@{
            Record = [pscustomobject]$record
            Process = $null
            StdoutTask = $null
            StderrTask = $null
        })
        Write-Output ("LAUNCH_FAILED|{0}|{1}" -f $exampleName, $record.launch_error)
    }
}

$remaining = [System.Collections.Generic.List[object]]::new()
foreach ($launch in $launches) {
    if ($launch.Process) {
        $remaining.Add($launch)
    }
}

while ($remaining.Count -gt 0) {
    foreach ($launch in @($remaining)) {
        if (-not $launch.Process.HasExited) {
            continue
        }

        $launch.Process.WaitForExit()
        $stdout = $launch.StdoutTask.GetAwaiter().GetResult()
        $stderr = $launch.StderrTask.GetAwaiter().GetResult()
        [System.IO.File]::WriteAllText($launch.Record.stdout, $stdout)
        [System.IO.File]::WriteAllText($launch.Record.stderr, $stderr)

        $combined = $stdout + "`n" + $stderr
        $nativeOutput = if ($launch.Record.native_log -and (Test-Path -LiteralPath $launch.Record.native_log)) {
            [System.IO.File]::ReadAllText($launch.Record.native_log)
        } else {
            ""
        }
        $launch.Record.ended_at = (Get-Date).ToString("o")
        $launch.Record.exit_code = $launch.Process.ExitCode
        $launch.Record.debugger_exit_code = if ($NativeDebug) { $launch.Record.exit_code } else { $null }
        $launch.Record.rust_backtrace = $combined -match "stack backtrace:|backtrace::|RUST_BACKTRACE"
        $launch.Record.native_exception = $nativeOutput -match "Exception|FAULTING_IP|Access violation|ntdll!|KERNELBASE!"
        $launch.Record.diagnostics = Get-DiagnosticDetails -StandardOutput $stdout -StandardError $stderr -NativeOutput $nativeOutput
        $launch.Record.status = if ($launch.Record.exit_code -eq 0 -and -not $launch.Record.native_exception) { "exited" } else { "crashed" }

        Write-Output ("EXITED|{0}|pid={1}|code={2}|status={3}|rust-backtrace={4}|native-exception={5}" -f `
            $launch.Record.name, $launch.Record.pid, $launch.Record.exit_code, $launch.Record.status,
            $launch.Record.rust_backtrace, $launch.Record.native_exception)
        $launch.Process.Dispose()
        $null = $remaining.Remove($launch)
    }

    if ($remaining.Count -gt 0) {
        Start-Sleep -Milliseconds 250
    }
}

$finalRecords = @($launches | ForEach-Object { $_.Record })
$summary = [ordered]@{
    started_at = $finalRecords[0].started_at
    ended_at = (Get-Date).ToString("o")
    profile = $profileName
    native_debug = $NativeDebug.IsPresent
    native_debugger = $debuggerPath
    total = $finalRecords.Count
    exited = @($finalRecords | Where-Object { $_.status -eq "exited" }).Count
    crashed = @($finalRecords | Where-Object { $_.status -eq "crashed" }).Count
    launch_failed = @($finalRecords | Where-Object { $_.status -eq "launch-failed" }).Count
    rust_backtraces = @($finalRecords | Where-Object { $_.rust_backtrace }).Count
    native_exceptions = @($finalRecords | Where-Object { $_.native_exception }).Count
    results = $finalRecords
}

$summaryJsonPath = Join-Path $runDirectory "summary.json"
$summaryTextPath = Join-Path $runDirectory "summary.txt"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryJsonPath

$textLines = [System.Collections.Generic.List[string]]::new()
$textLines.Add(("SUMMARY|total={0}|exited={1}|crashed={2}|launch-failed={3}|rust-backtraces={4}|native-exceptions={5}" -f `
    $summary.total, $summary.exited, $summary.crashed, $summary.launch_failed, $summary.rust_backtraces, $summary.native_exceptions))
foreach ($record in $finalRecords | Sort-Object name) {
    $textLines.Add(("{0}|status={1}|exit={2}|pid={3}|rust-backtrace={4}|native-exception={5}" -f `
        $record.name, $record.status, $record.exit_code, $record.pid, $record.rust_backtrace, $record.native_exception))
    foreach ($line in @($record.diagnostics.panic + $record.diagnostics.errors + $record.diagnostics.native_exception | Select-Object -Unique)) {
        if ($line) {
            $textLines.Add(("DETAIL|{0}|{1}" -f $record.name, $line.Trim()))
        }
    }
}
$textLines | Set-Content -LiteralPath $summaryTextPath

Write-Output ("SUMMARY_PATH|{0}" -f $summaryJsonPath)
Write-Output $textLines

if ($summary.crashed -gt 0 -or $summary.launch_failed -gt 0) {
    exit 1
}
