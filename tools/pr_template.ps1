param([string]$Branch = ("auto/fix-" + (Get-Date -Format yyyyMMdd-HHmm)))
$ErrorActionPreference = "Stop"

git checkout -b $Branch
git add .
git commit -m "chore(auto): apply code fixes and tests"
$diff = git diff HEAD~1 HEAD
$body = @"
## Summary
- Auto-applied patch via PLAN/PATCH
- Added/updated tests

## Diff


$diff

"@
$body | Out-File -Encoding UTF8 PR_BODY.md
Write-Host ("Branch={0}; PR_BODY.md ready (copy into GitHub PR)" -f $Branch) -ForegroundColor Green


