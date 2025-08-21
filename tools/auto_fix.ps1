param(
	[Parameter(Mandatory=$true)][string]$Path,
	[Parameter(Mandatory=$true)][string]$Function,
	[Parameter(Mandatory=$true)][string]$Intent,
	[string]$Port = "8765",
	[string]$TestPath,
	[switch]$DryRun
)
$ErrorActionPreference = "Stop"

function Get-FunctionSnippet {
	param([string]$Path,[string]$Func,[int]$Pad=20)
	# 안전: 파일에서 라인 단위로 읽되, 없으면 Raw 1200자 컷
	$lines = Get-Content -LiteralPath $Path -Encoding UTF8
	$hit = ($lines | Select-String -SimpleMatch "def $Func").LineNumber
	if(-not $hit){
		$raw = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
		return $raw.Substring(0, [Math]::Min(1200, $raw.Length))
	}
	$start=[Math]::Max(1,$hit-$Pad); $end=[Math]::Min($lines.Count,$hit+$Pad)
	return ($lines[($start-1)..($end-1)] -join "`n")
}

$src = Get-FunctionSnippet -Path $Path -Func $Function -Pad 30
$planBody = @{ intent=$Intent; paths=@(); code_paste=$src } | ConvertTo-Json -Depth 40 -Compress
$plan = Invoke-RestMethod "http://127.0.0.1:$Port/plan" -Method Post -ContentType "application/json; charset=utf-8" -Body $planBody -TimeoutSec 30
$planObj = if ($plan.plan -is [string]) { $plan.plan | ConvertFrom-Json } else { $plan.plan }

$fb = @{ hint="Return ONLY items of the edits array. No markdown."; reason="auto_fix" }
$patchBody = @{ plan=$planObj; feedback=$fb } | ConvertTo-Json -Depth 100 -Compress
$patch = Invoke-RestMethod "http://127.0.0.1:$Port/patch_smart" -Method Post -ContentType "application/json; charset=utf-8" -Body $patchBody -TimeoutSec 45

if(-not $patch.patch){
	Write-Error "PATCH_SMART failed"
	exit 1
}

# 적용 단계 (/apply)
$applyBody = @{ patch = $patch.patch; allowed_paths = @($Path); dry_run = [bool]$DryRun } | ConvertTo-Json -Depth 100 -Compress
$apply = Invoke-RestMethod "http://127.0.0.1:$Port/apply" -Method Post -ContentType "application/json" -Body $applyBody -TimeoutSec 40

# 결과 요약 및 디프 출력
Write-Host ("Result: applied={0} skipped={1} failed={2} dry_run={3}" -f (
    ($apply.applied | Measure-Object | Select-Object -ExpandProperty Count),
    ($apply.skipped  | Measure-Object | Select-Object -ExpandProperty Count),
    ($apply.failed   | Measure-Object | Select-Object -ExpandProperty Count),
    $apply.dry_run
))

if($apply.dry_run){
    Write-Host "`n==== DRY-RUN: no changes written ====" -ForegroundColor Yellow
} else {
    Write-Host "`n==== DIFF (working tree) ====" -ForegroundColor Yellow
    git --no-pager diff -- $Path | Write-Output
}

if($DryRun){ Write-Host "DryRun enabled. Skipping test/commit." -ForegroundColor Yellow; exit 0 }

# 간단 pytest 실행(있으면)
if ($DryRun) { exit 0 }

# 테스트 경로 기본값: <파일 디렉터리>\tests 존재 시 사용
if (-not $TestPath -or $TestPath.Trim() -eq "") {
	$defaultTests = Join-Path (Split-Path $Path -Parent) "tests"
	if (Test-Path $defaultTests) { $TestPath = $defaultTests }
}

$testsPassed = $true
if ($TestPath -and (Get-Command pytest -ErrorAction SilentlyContinue)) {
	Write-Host ("`nRunning pytest on {0}..." -f $TestPath) -ForegroundColor Cyan
	pytest -q -- $TestPath
	if ($LASTEXITCODE -ne 0) { $testsPassed = $false }
}

if ($testsPassed) {
	git add -- $Path
	git commit -m "auto-fix($Function): $Intent via PATCH_SMART" | Out-Null
	Write-Host "Committed." -ForegroundColor Green
} else {
	Write-Warning "Tests failed. Reverting changes to $Path."
	git restore --staged -- $Path 2>$null
	git checkout -- $Path
	exit 1
}


