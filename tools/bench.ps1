param(
  [int]$N = 20,
  [string]$Path = "examples\sample_py\app.py",
  [string]$FunctionName = "add",
  [string]$BaseHost = "127.0.0.1",
  [int]$Port = 8765
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-FunctionSnippet($Path,$Func,$Pad=20){
  $lines = Get-Content -Encoding UTF8 $Path
  $hit = ($lines | Select-String -SimpleMatch "def $Func").LineNumber
  if(-not $hit){ return (Get-Content -Raw -Encoding UTF8 $Path).Substring(0,[Math]::Min(1200,(Get-Content -Raw -Encoding UTF8 $Path).Length)) }
  $start=[Math]::Max(1,$hit-$Pad); $end=[Math]::Min($lines.Count,$hit+$Pad)
  return ($lines[($start-1)..($end-1)] -join "`n")
}

function Get-P50([double[]]$arr){
  $a = $arr | Where-Object { $_ -ne $null } | Sort-Object
  if ($a.Count -eq 0) { return $null }
  if ($a.Count % 2 -eq 1) { return [double]$a[([int](($a.Count-1)/2))] }
  else { return ([double]$a[($a.Count/2 - 1)] + [double]$a[($a.Count/2)]) / 2.0 }
}

$results = @()
for ($i=1; $i -le $N; $i++) {
  try {
    $src = Get-FunctionSnippet $Path $FunctionName 20
    $planBody = @{ intent="$FunctionName() 개선"; paths=@(); code_paste=$src } | ConvertTo-Json -Depth 40 -Compress
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $plan = Invoke-RestMethod "http://$($BaseHost):$Port/plan" -Method Post -ContentType 'application/json; charset=utf-8' -Body $planBody -TimeoutSec 30
    $sw.Stop(); $plan_ms = [int]$sw.Elapsed.TotalMilliseconds

    $planObj = if ($plan.plan -is [string]) { $plan.plan | ConvertFrom-Json } else { $plan.plan }
    $fb = @{ hint="Return ONLY items of the edits array. No markdown."; reason="bench" }
    $patchBody = @{ plan=$planObj; feedback=$fb } | ConvertTo-Json -Depth 100
    $sw.Restart()
    $patch = Invoke-RestMethod "http://$($BaseHost):$Port/patch_smart" -Method Post -ContentType 'application/json; charset=utf-8' -Body $patchBody -TimeoutSec 40
    $sw.Stop(); $patch_ms = [int]$sw.Elapsed.TotalMilliseconds

    $edits = 0
    if ($patch.patch -and $patch.patch.edits) { $edits = ($patch.patch.edits | Measure-Object).Count }
    elseif ($patch.edits) { $edits = ($patch.edits | Measure-Object).Count }
    $ok = $edits -gt 0
    $notes = if ($ok) { "ok" } else { "patch.edits 비어있음" }

    $results += [pscustomobject]@{ iter=$i; plan_ms=$plan_ms; patch_ms=$patch_ms; edits=$edits; ok=$ok; notes=$notes }
    Write-Host "[$i/$N] ok=$ok edits=$edits plan=${plan_ms}ms patch=${patch_ms}ms"
  } catch {
    $results += [pscustomobject]@{ iter=$i; plan_ms=$null; patch_ms=$null; edits=0; ok=$false; notes="error: $($_.Exception.Message)" }
    Write-Host "[$i/$N] ok=False edits=0 notes=error"
  }
}

$csvPath = "bench_results.csv"
$results | Export-Csv -NoTypeInformation -Path $csvPath -Encoding UTF8
Write-Host "CSV saved to $csvPath"

$planVals  = @($results.plan_ms)  | Where-Object { $_ -ne $null }
$patchVals = @($results.patch_ms) | Where-Object { $_ -ne $null }
$p50_plan  = Get-P50 ($planVals)
$p50_patch = Get-P50 ($patchVals)
$okCount   = (@($results | Where-Object { $_.ok })).Count
$totCount  = (@($results)).Count
$failCount = $totCount - $okCount
$err_rate  = if ($totCount -gt 0) { [math]::Round(($failCount / [double]$totCount) * 100, 2) } else { 0 }
$json_fail = 0
Write-Host "Done. p50_plan(ms)=$p50_plan p50_patch(ms)=$p50_patch json_fail%=$json_fail err%=$err_rate"

$hist = "$(Get-Date -Format 'yyyy-MM-dd HH:mm'),$p50_plan,$p50_patch,$json_fail,$err_rate"
Add-Content -Path "bench_history.csv" -Value $hist -Encoding UTF8



