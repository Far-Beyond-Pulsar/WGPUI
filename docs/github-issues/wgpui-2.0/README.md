# WGPUI 2.0 issue backlog

Each Markdown file in this directory is a GitHub issue source document. The
front matter contains the issue id, title, requested state, and labels; the
remaining Markdown is submitted as the issue body.

Use the filing script from the repository root:

```powershell
pwsh -File scripts/file-issues.ps1 `
  -Folder docs/github-issues/wgpui-2.0 `
  -CloseCompleted
```

When `-Repo` is omitted, the script targets the repository parsed from
`remote.origin`. Pass `-Repo OWNER/REPOSITORY` when intentionally filing in a
different repository. The earlier batch was mis-targeted to the upstream
`gpui-ce/gpui-ce`; those 80 issues were annotated and closed. The corrected
batch is filed on `Far-Beyond-Pulsar/WGPUI`.

Use `-DryRun` to validate front matter and duplicate titles without making
GitHub changes. The latest submission ledger is `results.json`.
