# tools/auto_fix.ps1  (PS5.1 / PS7 호환)
param(
  [Parameter(Mandatory=$true)][string]$Path,
  [Parameter(Mandatory=$true)][string]$Function,
  [Parameter(Mandatory=$true)][string]$Intent,
  [string]$Port = "8765",
  [switch]$DryRun,
  [string]$TestPath   # 선택: pytest를 특정 폴더/파일만 돌릴 때
)
$ErrorActionPreference = "Stop"

function Get-FunctionSnippet([string]$Path,[string]$Func,[int]$Pad=30){
  $lines = Get-Content -Encoding UTF8 -LiteralPath $Path
  $hit = ($lines | Select-String -SimpleMatch "def $Func").LineNumber
  if(-not $hit){
    $raw = Get-Content -Raw -Encoding UTF8 -LiteralPath $Path
    $max = [math]::Min(1200, $raw.Length)
    return $raw.Substring(0,$max)
  }
  $start=[Math]::Max(1,$hit-$Pad); $end=[Math]::Min($lines.Count,$hit+$Pad)
  return ($lines[($start-1)..($end-1)] -join "`n")
}

# 1) 스니펫 준비
$src = Get-FunctionSnippet -Path $Path -Func $Function -Pad 30

# 2) PLAN 호출 (파일 경로 포함, 의도에 함수명 프리픽스)
$planBody = @{
  intent     = ("{0}: {1}" -f $Function, $Intent)
  paths      = @($Path)
  code_paste = $src
} | ConvertTo-Json -Depth 40 -Compress

$plan = Invoke-RestMethod "http://127.0.0.1:$Port/plan" `
        -Method Post -ContentType "application/json; charset=utf-8" `
        -Body $planBody -TimeoutSec 30

$planObj = if ($plan.plan -is [string]) { $plan.plan | ConvertFrom-Json } else { $plan.plan }

# 3) PATCH_SMART 호출
$fb = @{ hint="Return ONLY items of the edits array. No markdown."; reason="auto_fix" }
$patchBody = @{ plan=$planObj; feedback=$fb } | ConvertTo-Json -Depth 100 -Compress

$patch = Invoke-RestMethod "http://127.0.0.1:$Port/patch_smart" `
         -Method Post -ContentType "application/json; charset=utf-8" `
         -Body $patchBody -TimeoutSec 45

# 서버 응답 호환 처리: {patch:{edits:[]}} 형태 또는 {edits:[]} 형태 모두 지원
$patchData = $null
if ($patch -and ($patch.PSObject.Properties.Name -contains 'patch') -and $patch.patch) {
  $patchData = $patch.patch
} else {
  $patchData = $patch
}
if (-not $patchData) { Write-Error "PATCH_SMART failed: empty response" }

# 4) DIFF 출력 (있으면)
if($patch.PSObject.Properties.Name -contains 'diff_path' -and $patch.diff_path -and (Test-Path -LiteralPath $patch.diff_path)){
  Write-Host "`n==== DIFF ====" -ForegroundColor Yellow
  Get-Content -Encoding UTF8 -LiteralPath $patch.diff_path | Write-Output
}
$editCount = 0; if ($patchData.PSObject.Properties.Name -contains 'edits' -and $patchData.edits){ $editCount = ($patchData.edits | Measure-Object).Count }
Write-Host ("Result: edits={0}" -f $editCount)
if ($editCount -lt 1) { Write-Error "No edits returned from PATCH_SMART" }

# /apply 단계 (드라이런 선택)
$applyBody = @{ patch = $patchData; allowed_paths = @($Path); dry_run = [bool]$DryRun } | ConvertTo-Json -Depth 100 -Compress
$apply = Invoke-RestMethod "http://127.0.0.1:$Port/apply" -Method Post -ContentType "application/json" -Body $applyBody -TimeoutSec 40
${appliedCount} = ($apply.applied | Measure-Object | Select-Object -ExpandProperty Count)
${skippedCount} = ($apply.skipped  | Measure-Object | Select-Object -ExpandProperty Count)
${failedCount}  = ($apply.failed   | Measure-Object | Select-Object -ExpandProperty Count)
Write-Host ("Apply: applied={0}, skipped={1}, failed={2}, dry_run={3}" -f ${appliedCount}, ${skippedCount}, ${failedCount}, $apply.dry_run)
if (-not $apply.dry_run -and ${appliedCount} -lt 1 -and ${failedCount} -gt 0) {
  Write-Error "Apply failed. Aborting before tests/commit."
}
if (-not $apply.dry_run) {
  Write-Host "`n==== DIFF (working tree) ====" -ForegroundColor Yellow
  git --no-pager diff -- $Path | Write-Output
}

if($DryRun){ Write-Host "DryRun enabled. Skipping pytest/commit."; exit 0 }

# 5) pytest 실행(있으면)
if (Get-Command pytest -ErrorAction SilentlyContinue) {
  $repo = (& git rev-parse --show-toplevel 2>$null)
  if ($LASTEXITCODE -eq 0 -and $repo) {
    $workdir = $repo
  } else {
    $workdir = (Get-Location).Path
  }
  Push-Location $workdir
  try {
    $testArg = $null
    if ($TestPath -and $TestPath.Trim().Length -gt 0) { $testArg = (Resolve-Path -LiteralPath $TestPath).Path }

    if ($null -ne $testArg) {
      Write-Host ("`nRunning pytest in {0} on {1}" -f $workdir, $testArg) -ForegroundColor Cyan
      & pytest -q --maxfail=1 --disable-warnings --color=yes $testArg
    } else {
      Write-Host ("`nRunning pytest in {0} (all)" -f $workdir) -ForegroundColor Cyan
      & pytest -q --maxfail=1 --disable-warnings --color=yes
    }
  } catch {
    Write-Warning ("pytest failed: {0}" -f $_.Exception.Message)
  } finally {
    Pop-Location
  }
} else {
  Write-Host "pytest not found. Skipping test step." -ForegroundColor DarkYellow
}

# 6) 테스트 결과에 따른 커밋/롤백
if ($LASTEXITCODE -eq 0) {
  git add -- $Path
  git commit -m ("auto-fix({0}): {1} via PATCH_SMART" -f $Function, $Intent) | Out-Null
  Write-Host "Committed." -ForegroundColor Green
} else {
  Write-Warning "Tests failed. Reverting changes."
  git restore --staged -- $Path 2>$null
  git checkout -- $Path
  exit 1
}
