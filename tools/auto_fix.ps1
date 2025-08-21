param(
	[Parameter(Mandatory=$true)][string]$Path,
	[Parameter(Mandatory=$true)][string]$Function,
	[Parameter(Mandatory=$true)][string]$Intent,
	[string]$Port = "8765",
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

# 디프(서버에서 details를 반환하지 않는 경우, 로컬 git diff로 대체)
Write-Host "`n==== DIFF (git status) ====" -ForegroundColor Yellow
git status --porcelain

Write-Host ("Result: edits={0}" -f ($patch.patch.edits | Measure-Object | Select-Object -ExpandProperty Count))

if($DryRun){ Write-Host "DryRun enabled. Skipping apply/test/commit." -ForegroundColor Yellow; exit 0 }

# 간단 pytest 실행(있으면)
if (Get-Command pytest -ErrorAction SilentlyContinue) {
	try { Write-Host "`nRunning pytest..." -ForegroundColor Cyan; pytest -q } catch { Write-Warning "pytest failed: $($_.Exception.Message)" }
}

# 자동 커밋
git add -- $Path
git commit -m "auto-fix($Function): $Intent via PATCH_SMART" | Out-Null
Write-Host "Committed." -ForegroundColor Green


