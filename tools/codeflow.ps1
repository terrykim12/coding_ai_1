# PowerShell 5.1 호환 코드 흐름 유틸리티 (함수 스니펫 기반 호출)
# Set-StrictMode와 ErrorAction을 적극 사용. 삼항 연산자 불가 환경 고려 [[PS5.1]]

Set-StrictMode -Version Latest

function Get-HealthyPort {
  param([int[]]$Ports = @(8765,8777))
  foreach ($p in $Ports) {
    try {
      $r = Invoke-RestMethod -Uri "http://127.0.0.1:$p/health" -TimeoutSec 3 -ErrorAction Stop
      if ($r.status -eq 'healthy') { return $p }
    } catch {}
  }
  throw "서버를 찾을 수 없습니다 (시도: $($Ports -join ', '))."
}

function Get-PyFunctionSnippet {
  param(
    [Parameter(Mandatory)] [string]$Path,
    [Parameter(Mandatory)] [string]$FunctionName
  )
  if (-not (Test-Path $Path)) { throw "파일 없음: $Path" }
  $lines = Get-Content -Path $Path
  $start = -1
  for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match ('^[\t ]*def\s+' + [regex]::Escape($FunctionName) + '\s*\(')) { $start = $i; break }
  }
  if ($start -lt 0) { throw "함수 스니펫을 찾을 수 없습니다: $FunctionName @ $Path" }
  $end = $lines.Count
  for ($j = $start + 1; $j -lt $lines.Count; $j++) {
    if ($lines[$j] -match '^\S') { $end = $j; break }
  }
  return ($lines[$start..($end-1)] -join "`r`n")
}

function Invoke-CodeEdit {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)] [string]$Intent,
    [Parameter(Mandatory)] [string]$Path,              # 예: examples\sample_py\app.py
    [string]$FunctionName,                             # 예: add
    [string]$AnchorRegex,                              # 예: 'def add\('
    [string]$ProjectRoot = (Get-Location).Path,
    [int]$TimeoutPlan = 120, [int]$TimeoutPatch = 180, [int]$TimeoutApply = 120, [int]$TimeoutTest = 180
  )
  $port = Get-HealthyPort
  $abs = (Resolve-Path $Path).Path
  # PowerShell 5.1(.NET Framework)에는 GetRelativePath가 없어 수동 계산
  $proj = (Resolve-Path $ProjectRoot).Path
  if ($abs.ToLower().StartsWith($proj.ToLower())) {
    $rel = $abs.Substring($proj.Length).TrimStart('\')
  } else {
    $rel = $abs
  }

  # 1) code_paste는 함수 스니펫 우선
  $code_paste = $null
  if ($FunctionName) {
    $code_paste = Get-PyFunctionSnippet -Path $abs -FunctionName $FunctionName
  } else {
    $code_paste = Get-Content -Raw $abs     # 최후수단: 전체 파일
  }

  # 2) PLAN
  $planReq = @{ intent=$Intent; paths=@($rel); code_paste=$code_paste } | ConvertTo-Json -Depth 40
  $plan = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/plan" `
           -ContentType "application/json" -Body $planReq -TimeoutSec $TimeoutPlan
  $planObj = $null
  if ($plan.plan -is [string]) { $planObj = $plan.plan | ConvertFrom-Json } else { $planObj = $plan.plan }
  if (-not $planObj) { throw "plan 비어있음" }

  # 3) PATCH_SMART (형식 힌트 포함)
  $hint = "Return ONLY the items of the edits array (each item is a JSON object). No code fences or markdown. Use only actions: insert_before, insert_after, replace_range, delete_range."
  $psBody = @{ plan=$planObj; feedback=@{ hint=$hint; reason="strict JSON required" } } | ConvertTo-Json -Depth 100
  $patchResp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/patch_smart" `
                -ContentType "application/json" -Body $psBody -TimeoutSec $TimeoutPatch -ErrorAction Stop
  $patchObj = $null
  if ($patchResp.patch -is [string]) { $patchObj = $patchResp.patch | ConvertFrom-Json } else { $patchObj = $patchResp.patch }
  if (-not $patchObj) { throw "patch 비어있음" }
  if (-not $patchObj.edits -or $patchObj.edits.Count -eq 0) { throw "patch.edits 비어있음" }

  # 4) APPLY (dry-run → real)
  $allowed = @(Split-Path $rel)  # 파일이 속한 디렉터리만 허용
  $applyDry  = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/apply" `
                 -ContentType "application/json" `
                 -Body (@{ patch=$patchObj; allowed_paths=$allowed; dry_run=$true } | ConvertTo-Json -Depth 100) `
                 -TimeoutSec $TimeoutApply
  $applyReal = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/apply" `
                 -ContentType "application/json" `
                 -Body (@{ patch=$patchObj; allowed_paths=$allowed; dry_run=$false } | ConvertTo-Json -Depth 100) `
                 -TimeoutSec $TimeoutApply

  # 5) TEST
  $test = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/test" -ContentType "application/json" `
            -Body (@{ paths=@($rel | Split-Path -Parent) } | ConvertTo-Json) -TimeoutSec $TimeoutTest

  return [pscustomobject]@{
    port       = $port
    plan_id    = $plan.plan_id
    edits      = $patchObj.edits.Count
    applyDry   = $applyDry.details
    applyReal  = $applyReal.details
    test       = $test.summary
  }
}

# 사용 예시
# . .\tools\codeflow.ps1
# Invoke-CodeEdit -Intent "add()에 음수 방지 추가" -Path "examples\sample_py\app.py" -FunctionName "add" |
#   ConvertTo-Json -Depth 60



