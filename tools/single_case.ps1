param(
  [int]$Port = 8765,
  [string]$Path = "examples\sample_py\app.py",
  [string]$FunctionName = "add"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-JsonPost {
  param(
    [Parameter(Mandatory)] [string]$Uri,
    [Parameter(Mandatory)] $Object,
    [int]$TimeoutSec = 240
  )
  $json = $Object | ConvertTo-Json -Depth 100
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
  return Invoke-RestMethod -Method Post -Uri $Uri -ContentType "application/json; charset=utf-8" -Body $bytes -TimeoutSec $TimeoutSec
}

if (-not (Test-Path $Path)) { throw "파일 없음: $Path" }
$src = Get-Content -Raw $Path
$pat = '(?smi)^[ \t]*def[ \t]+' + [regex]::Escape($FunctionName) + '\(.*?\):\s*\r?\n(?:[ \t]+.*\r?\n)*'
$m = [regex]::Match($src,$pat)
if ($m.Success) { $snippet = $m.Value } else { $snippet = $src }

# PLAN
$t0 = Get-Date
$planReq = @{ intent = "add() 음수 방지"; paths = @("examples/sample_py"); code_paste = $snippet }
$plan = Invoke-JsonPost -Uri ("http://127.0.0.1:" + $Port + "/plan") -Object $planReq -TimeoutSec 240
$t1 = Get-Date
Write-Output ("PLAN ms=" + ([int]($t1 - $t0).TotalMilliseconds))

$planObj = $null
if ($plan.plan -is [string]) { $planObj = $plan.plan | ConvertFrom-Json } else { $planObj = $plan.plan }
if (-not $planObj) { throw "plan 비어있음" }

# PATCH_SMART (실패 시 /patch)
$hint = "Return ONLY the items of the edits array (each item is a JSON object). No code fences or markdown. Use only actions: insert_before, insert_after, replace_range, delete_range."
$body = @{ plan = $planObj; feedback = @{ hint = $hint; reason = "log only" } }

$t2 = Get-Date
try {
  $patch = Invoke-JsonPost -Uri ("http://127.0.0.1:" + $Port + "/patch_smart") -Object $body -TimeoutSec 300
  $t3 = Get-Date
  Write-Output ("PATCH_SMART ms=" + ([int]($t3 - $t2).TotalMilliseconds))
} catch {
  $t3 = Get-Date
  Write-Output ("PATCH_SMART failed after ms=" + ([int]($t3 - $t2).TotalMilliseconds) + "; trying /patch")
  $t4 = Get-Date
  $patch = Invoke-JsonPost -Uri ("http://127.0.0.1:" + $Port + "/patch") -Object $body -TimeoutSec 300
  $t5 = Get-Date
  Write-Output ("PATCH ms=" + ([int]($t5 - $t4).TotalMilliseconds))
}

$po = $null
if ($patch.patch -is [string]) { $po = $patch.patch | ConvertFrom-Json } else { $po = $patch.patch }
if ($po -and $po.edits) {
  Write-Output ("edits=" + $po.edits.Count)
} else {
  Write-Output "edits=0"
}


