param(
  [int]$Port = 8765,
  [string]$Path = "examples\sample_py\app.py",
  [string]$FunctionName = "add",
  [int]$PlanTimeout = 180,
  [int]$PatchTimeout = 240
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. "$PSScriptRoot\utf8.ps1" 2>$null

function Invoke-JsonPost {
  param(
    [Parameter(Mandatory)] [string]$Uri,
    [Parameter(Mandatory)] $Object,
    [int]$TimeoutSec = 180
  )
  $json = $Object | ConvertTo-Json -Depth 100
  $bytes = [Text.Encoding]::UTF8.GetBytes($json)
  return Invoke-RestMethod -Method Post -Uri $Uri -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec $TimeoutSec
}

# 헬스
try {
  $h = Invoke-RestMethod -Uri ("http://127.0.0.1:"+$Port+"/health") -TimeoutSec 6
  Write-Host ("HEALTH: status="+$h.status+" model_loaded="+$h.model_loaded+" use_4bit="+$h.use_4bit)
} catch { Write-Host "HEALTH: unavailable" }

if (-not (Test-Path $Path)) { throw "파일 없음: $Path" }
$src = Get-Content -Raw $Path
$pat = '(?smi)^[ \t]*def[ \t]+' + [regex]::Escape($FunctionName) + '\(.*?\):\s*\r?\n(?:[ \t]+.*\r?\n)*'
$m = [regex]::Match($src,$pat)
$snippet = if ($m.Success) { $m.Value } else { $src }

# PLAN
$t0 = Get-Date
$planReq = @{ intent = "add() 음수 방지"; paths = @("examples/sample_py"); code_paste = $snippet }
$plan = Invoke-JsonPost -Uri ("http://127.0.0.1:"+$Port+"/plan") -Object $planReq -TimeoutSec $PlanTimeout
$t1 = Get-Date
Write-Host ("PLAN: " + [int]($t1-$t0).TotalSeconds + "s")

$planObj = if ($plan.plan -is [string]) { $plan.plan | ConvertFrom-Json } else { $plan.plan }
if (-not $planObj) { throw "plan 비어있음" }

# PATCH_SMART → 실패 시 /patch
$hint = "Return ONLY the items of the edits array (each item is a JSON object). No fences/markdown. Use only: insert_before, insert_after, replace_range, delete_range."
$body = @{ plan = $planObj; feedback = @{ hint = $hint; reason = "log only" } }

$t2 = Get-Date
try {
  $patch = Invoke-JsonPost -Uri ("http://127.0.0.1:"+$Port+"/patch_smart") -Object $body -TimeoutSec $PatchTimeout
  $t3 = Get-Date
  Write-Host ("PATCH_SMART: " + [int]($t3-$t2).TotalSeconds + "s")
} catch {
  $t3 = Get-Date
  Write-Host ("PATCH_SMART failed after " + [int]($t3-$t2).TotalSeconds + "s → trying /patch")
  $t4 = Get-Date
  $patch = Invoke-JsonPost -Uri ("http://127.0.0.1:"+$Port+"/patch") -Object $body -TimeoutSec $PatchTimeout
  $t5 = Get-Date
  Write-Host ("PATCH: " + [int]($t5-$t4).TotalSeconds + "s")
}

$po = if ($patch.patch -is [string]) { $patch.patch | ConvertFrom-Json } else { $patch.patch }
if ($po -and $po.edits) { Write-Host ("edits="+$po.edits.Count) } else { Write-Host "edits=0" }


