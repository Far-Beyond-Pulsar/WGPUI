[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Folder,
    [string] $Repo,
    [switch] $DryRun,
    [switch] $CloseCompleted,
    [switch] $ApplyLabels,
    [string] $Output = "issue-results.json",
    [int] $Limit = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-FrontMatterIssue {
    param([string] $Path)

    $content = Get-Content -LiteralPath $Path -Raw
    $match = [regex]::Match($content, "(?s)\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)\z")
    if (-not $match.Success) {
        throw "Issue file '$Path' must start with YAML front matter delimited by --- lines."
    }

    $fields = @{}
    foreach ($line in ($match.Groups[1].Value -split "\r?\n")) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $separator = $line.IndexOf(":")
        if ($separator -lt 1) {
            throw "Invalid front matter line '$line' in '$Path'."
        }
        $key = $line.Substring(0, $separator).Trim()
        $value = $line.Substring($separator + 1).Trim()
        $fields[$key] = $value.Trim([char]39, [char]34)
    }

    if (-not $fields.ContainsKey("title") -or [string]::IsNullOrWhiteSpace($fields["title"])) {
        throw "Issue file '$Path' is missing title front matter."
    }

    $state = if ($fields.ContainsKey("state")) { $fields["state"].ToLowerInvariant() } else { "open" }
    if ($state -notin @("open", "closed")) {
        throw "Issue file '$Path' has unsupported state '$state'; use open or closed."
    }

    $labels = @()
    if ($fields.ContainsKey("labels") -and -not [string]::IsNullOrWhiteSpace($fields["labels"])) {
        $labels = @($fields["labels"] -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    }

    [pscustomobject]@{
        Path = (Resolve-Path -LiteralPath $Path).Path
        Id = if ($fields.ContainsKey("id")) { $fields["id"] } else { [IO.Path]::GetFileNameWithoutExtension($Path) }
        Title = $fields["title"]
        State = $state
        Labels = $labels
        Body = $match.Groups[2].Value.Trim()
    }
}

function Get-OriginRepository {
    $remote = (& git config --get remote.origin.url 2>$null).Trim()
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($remote)) {
        $match = [regex]::Match($remote, "github\.com[:/]([^/\s]+/[^/\s]+?)(?:\.git)?$")
        if ($match.Success) {
            return $match.Groups[1].Value
        }
    }
    return $null
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "The gh CLI is required."
}

$folderPath = (Resolve-Path -LiteralPath $Folder).Path
$issueFiles = @(Get-ChildItem -LiteralPath $folderPath -Filter "*.md" -File | Sort-Object Name)
if ($Limit -gt 0) {
    $issueFiles = @($issueFiles | Select-Object -First $Limit)
}
if ($issueFiles.Count -eq 0) {
    throw "No Markdown issue files were found in '$folderPath'."
}

$issues = @($issueFiles | ForEach-Object { Get-FrontMatterIssue -Path $_.FullName })
$duplicateTitles = @($issues | Group-Object Title | Where-Object Count -gt 1)
if ($duplicateTitles.Count -gt 0) {
    throw "Duplicate issue titles found: $($duplicateTitles.Name -join ', ')"
}

if ([string]::IsNullOrWhiteSpace($Repo)) {
    $Repo = Get-OriginRepository
    if ([string]::IsNullOrWhiteSpace($Repo)) {
        throw "Could not determine the repository from remote.origin. Pass -Repo OWNER/REPOSITORY explicitly."
    }
}

$existing = @{}
if (-not $DryRun) {
    $existingJson = & gh issue list --repo $Repo --state all --limit 1000 --json number,title,state,url
    if ($LASTEXITCODE -ne 0) {
        throw "Could not list existing issues for '$Repo'."
    }
    foreach ($item in (@($existingJson | ConvertFrom-Json))) {
        $existing[$item.title] = $item
    }
}

$results = [System.Collections.Generic.List[object]]::new()
foreach ($issue in $issues) {
    if ($existing.ContainsKey($issue.Title)) {
        $current = $existing[$issue.Title]
        $results.Add([pscustomobject]@{
                id = $issue.Id
                title = $issue.Title
                action = "skipped-existing"
                number = $current.number
                state = $current.state
                url = $current.url
            })
        continue
    }

    if ($DryRun) {
        $results.Add([pscustomobject]@{
                id = $issue.Id
                title = $issue.Title
                action = "dry-run"
                requestedState = $issue.State
                labels = $issue.Labels
            })
        continue
    }

    $arguments = @("issue", "create", "--repo", $Repo, "--title", $issue.Title, "--body", $issue.Body)
    if ($ApplyLabels -and $issue.Labels.Count -gt 0) {
        foreach ($label in $issue.Labels) {
            $arguments += @("--label", $label)
        }
    }
    $createdOutput = & gh @arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create '$($issue.Title)': $($createdOutput -join "`n")"
    }
    $url = ($createdOutput -join "`n") | Select-String -Pattern "https://github\.com/[^\s]+/issues/\d+" -AllMatches | ForEach-Object { $_.Matches.Value } | Select-Object -Last 1
    if ([string]::IsNullOrWhiteSpace($url)) {
        throw "gh did not return an issue URL for '$($issue.Title)'. Output: $($createdOutput -join "`n")"
    }
    $number = [int]([regex]::Match($url, "/issues/(\d+)$").Groups[1].Value)
    $state = "OPEN"
    $action = "created"

    if ($issue.State -eq "closed") {
        if (-not $CloseCompleted) {
            $action = "created-open-requested-closed"
        } else {
            $closedOutput = & gh issue close $number --repo $Repo --comment "Marked completed in $([IO.Path]::GetFileName($issue.Path)); implementation is present in the referenced commit(s)." 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw "Created issue #$number but failed to close it: $($closedOutput -join "`n")"
            }
            $state = "CLOSED"
            $action = "created-and-closed"
        }
    }

    $existing[$issue.Title] = [pscustomobject]@{ number = $number; state = $state; url = $url }
    $results.Add([pscustomobject]@{
            id = $issue.Id
            title = $issue.Title
            action = $action
            number = $number
            state = $state
            url = $url
        })
    Write-Output "$action #$number $($issue.Id) $url"
}

$results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Output -Encoding utf8
Write-Output "Wrote $($results.Count) result records to $Output"
