param(
    [switch]$Offline,
    [switch]$Locked,
    [string]$OutputDirectory = "",
    [switch]$KeepRawOutput
)

$ErrorActionPreference = "Stop"

$manifestPath = Join-Path $PSScriptRoot "..\crates\wgpui-examples-2\Cargo.toml"
$manifestText = Get-Content -Raw -Path $manifestPath
$reportDirectory = if ($OutputDirectory) {
    $OutputDirectory
} else {
    Join-Path $PSScriptRoot "..\target\examples-2-analysis"
}

New-Item -ItemType Directory -Force -Path $reportDirectory | Out-Null
$rawDirectory = Join-Path $reportDirectory "raw"
New-Item -ItemType Directory -Force -Path $rawDirectory | Out-Null

$entries = [System.Collections.Generic.List[object]]::new()
$entryPattern = '(?ms)^\[\[example\]\]\s*\r?\nname\s*=\s*"([^"]+)"\s*\r?\npath\s*=\s*"([^"]+)"'
foreach ($match in [regex]::Matches($manifestText, $entryPattern)) {
    $entries.Add([pscustomobject]@{
        Name = $match.Groups[1].Value
        Path = $match.Groups[2].Value
    })
}

if ($entries.Count -eq 0) {
    throw "No [[example]] entries were found in $manifestPath"
}

$cargoArguments = @(
    "check",
    "--manifest-path", $manifestPath,
    "--message-format=json"
)
if ($Offline) { $cargoArguments += "--offline" }
if ($Locked) { $cargoArguments += "--locked" }

function Get-NormalizedMessage([string]$message) {
    # Locations and generated dependency paths do not identify an API gap.
    $normalized = $message -replace '(?i)[A-Z]:\\[^\s:]+', '<path>'
    $normalized = $normalized -replace '(?i)(?:[A-Za-z0-9_.-]+[\\/])+[^\s:]+\.rs', '<source>'
    $normalized = $normalized -replace '\b\d+:\d+\b', '<location>'
    $normalized = $normalized -replace '\s+', ' '
    return $normalized.Trim()
}

function Get-Category([string]$code, [string]$message) {
    if ($message -match "cannot find module or crate|unresolved module or unlinked crate|bytemuck|\bwgpu\b") {
        return "dependency-or-low-level-gpu-surface"
    }
    if ($code -in @("E0432", "E0433", "E0405", "E0412", "E0425")) {
        return "missing-native-export-or-type"
    }
    if ($code -eq "E0599") { return "missing-native-method" }
    if ($code -in @("E0277", "E0308")) { return "native-signature-or-trait-mismatch" }
    if ($code -eq "E0554") { return "macro-or-feature-surface" }
    if ($message -match "cannot find derive macro|proc-macro|attribute") { return "macro-or-feature-surface" }
    return "other-compile-error"
}

$allDiagnostics = [System.Collections.Generic.List[object]]::new()
$exampleResults = [System.Collections.Generic.List[object]]::new()

for ($index = 0; $index -lt $entries.Count; $index++) {
    $entry = $entries[$index]
    Write-Output ("[{0}/{1}] {2}" -f ($index + 1), $entries.Count, $entry.Name)

    $output = @(& cargo @cargoArguments --example $entry.Name 2>&1)
    $exitCode = $LASTEXITCODE
    if ($KeepRawOutput) {
        $rawPath = Join-Path $rawDirectory ("{0}.jsonl" -f $entry.Name)
        $output | Set-Content -Path $rawPath
    }

    $exampleDiagnostics = [System.Collections.Generic.List[object]]::new()
    foreach ($line in $output) {
        $text = [string]$line
        if (-not $text.TrimStart().StartsWith("{")) { continue }
        try {
            $messageObject = $text | ConvertFrom-Json
        } catch {
            continue
        }
        if ($messageObject.reason -ne "compiler-message") { continue }
        $diagnostic = $messageObject.message
        if ($diagnostic.level -ne "error") { continue }
        $code = if ($diagnostic.code -and $diagnostic.code.code) { $diagnostic.code.code } else { "none" }
        $message = [string]$diagnostic.message
        $key = "{0}|{1}" -f $code, (Get-NormalizedMessage $message)
        $item = [pscustomobject]@{
            Example = $entry.Name
            Source = $entry.Path
            Code = $code
            Message = $message
            Key = $key
            Category = Get-Category $code $message
        }
        $exampleDiagnostics.Add($item)
        $allDiagnostics.Add($item)
    }

    $exampleResults.Add([pscustomobject]@{
        Name = $entry.Name
        Path = $entry.Path
        Passed = ($exitCode -eq 0)
        ExitCode = $exitCode
        ErrorCount = $exampleDiagnostics.Count
        Errors = @($exampleDiagnostics)
    })
}

$groups = $allDiagnostics | Group-Object Key | Sort-Object -Property Count, Name -Descending
$consolidated = foreach ($group in $groups) {
    $first = $group.Group[0]
    [pscustomobject]@{
        Occurrences = $group.Count
        Examples = @($group.Group | Select-Object -ExpandProperty Example -Unique | Sort-Object)
        Category = $first.Category
        Code = $first.Code
        Message = $first.Message
    }
}

$jsonReport = [pscustomobject]@{
    GeneratedAt = (Get-Date).ToUniversalTime().ToString("o")
    TotalExamples = $entries.Count
    PassedExamples = @($exampleResults | Where-Object Passed).Count
    FailedExamples = @($exampleResults | Where-Object { -not $_.Passed }).Count
    UniqueErrors = @($consolidated).Count
    Examples = @($exampleResults)
    Errors = @($consolidated)
}
$jsonPath = Join-Path $reportDirectory "report.json"
$jsonReport | ConvertTo-Json -Depth 8 | Set-Content -Path $jsonPath

$markdown = [System.Text.StringBuilder]::new()
[void]$markdown.AppendLine("# WGPUI 2.0 examples compile analysis")
[void]$markdown.AppendLine("")
[void]$markdown.AppendLine(("- Examples: **{0}**" -f $jsonReport.TotalExamples))
[void]$markdown.AppendLine(("- Passed: **{0}**" -f $jsonReport.PassedExamples))
[void]$markdown.AppendLine(("- Failed: **{0}**" -f $jsonReport.FailedExamples))
[void]$markdown.AppendLine(("- Unique normalized errors: **{0}**" -f $jsonReport.UniqueErrors))
[void]$markdown.AppendLine("")
[void]$markdown.AppendLine("## Consolidated errors")
[void]$markdown.AppendLine("")

if (@($consolidated).Count -eq 0) {
    [void]$markdown.AppendLine("No compiler errors were reported.")
} else {
    foreach ($errorGroup in $consolidated) {
        $exampleList = ($errorGroup.Examples -join ", ")
        [void]$markdown.AppendLine(("### {0} — {1} ({2} occurrences)" -f $errorGroup.Code, $errorGroup.Category, $errorGroup.Occurrences))
        [void]$markdown.AppendLine("")
        [void]$markdown.AppendLine(('`{0}`' -f $errorGroup.Message.Replace('`', '\`')))
        [void]$markdown.AppendLine("")
        [void]$markdown.AppendLine(("Examples: {0}" -f $exampleList))
        [void]$markdown.AppendLine("")
    }
}

[void]$markdown.AppendLine("## Per-example status")
[void]$markdown.AppendLine("")
[void]$markdown.AppendLine("| Example | Result | Errors | Source |")
[void]$markdown.AppendLine("|---|---:|---:|---|")
foreach ($result in $exampleResults) {
    $status = if ($result.Passed) { "PASS" } else { "FAIL" }
    [void]$markdown.AppendLine(("| `{0}` | {1} | {2} | `{3}` |" -f $result.Name, $status, $result.ErrorCount, $result.Path))
}

$markdownPath = Join-Path $reportDirectory "report.md"
$markdown.ToString() | Set-Content -Path $markdownPath

Write-Output ""
Write-Output ("SUMMARY|total={0}|passed={1}|failed={2}|unique-errors={3}" -f $jsonReport.TotalExamples, $jsonReport.PassedExamples, $jsonReport.FailedExamples, $jsonReport.UniqueErrors)
Write-Output ("REPORT|markdown={0}|json={1}" -f $markdownPath, $jsonPath)

if ($jsonReport.FailedExamples -gt 0) { exit 1 }
