param([int]$Repeat = 3)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-HealthyPort {
  param([int[]]$Ports=@(8765,8777))
  foreach ($p in $Ports) {
    try {
      $r = Invoke-RestMethod -Uri "http://127.0.0.1:$p/health" -TimeoutSec 3 -ErrorAction Stop
      if ($r.status -eq 'healthy') { return $p }
    } catch {}
  }
  throw "서버를 찾을 수 없습니다 (8765/8777)."
}

function Get-PySnippet([string]$Path,[string]$Func){
  if (-not (Test-Path $Path)) { throw "파일 없음: $Path" }
  $lines = Get-Content -Path $Path
  $start = -1
  for ($i=0; $i -lt $lines.Count; $i++){
    if ($lines[$i] -match ('^[\t ]*def\s+' + [regex]::Escape($Func) + '\s*\(')) { $start = $i; break }
  }
  if ($start -lt 0) { throw "스니펫 없음: $Func @ $Path" }
  $end = $lines.Count
  for ($j=$start+1; $j -lt $lines.Count; $j++){
    if ($lines[$j] -match '^\S'){ $end = $j; break }
  }
  return ($lines[$start..($end-1)] -join "`r`n")
}

$port = Get-HealthyPort
$path = "examples\sample_py\app.py"

# 파일을 건드리지 않는 의도들(적용/테스트 없이 패치 제안만)
$intents = @(
  "add()에 입력 타입검사 추가 (int만 허용)",
  "divide()에 0 나누기 방지 로직과 명확한 에러 메시지 추가",
  "factorial()에 음수 입력 방지 및 ValueError 메시지 보강",
  "is_prime()에 2 미만 처리 보강 및 빠른 return",
  "fibonacci()에 n==1 처리 보강 및 입력검증 추가"
)

$funcMap = @{
  "add" = "add"
  "divide" = "divide"
  "factorial" = "factorial"
  "is_prime" = "is_prime"
  "fibonacci" = "fibonacci"
}

function Invoke-JsonPost {
  param(
    [Parameter(Mandatory)] [string]$Uri,
    [Parameter(Mandatory)] $Object,
    [int]$TimeoutSec = 120
  )
  $json = $Object | ConvertTo-Json -Depth 100
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
  return Invoke-RestMethod -Method Post -Uri $Uri -ContentType "application/json; charset=utf-8" -Body $bytes -TimeoutSec $TimeoutSec
}

function Select-FunctionForIntent([string]$intent){
  foreach($name in @('add','divide','factorial','is_prime','fibonacci')){
    if ($intent -match [regex]::Escape($name) ) { return $name }
    if ($intent -match ([regex]::Escape($name) + '\(')) { return $name }
  }
  return 'add'
}

for($rep=1; $rep -le $Repeat; $rep++){
  foreach($intent in $intents){
    $fn = Select-FunctionForIntent $intent
    try{
      $snippet = Get-PySnippet -Path $path -Func $fn
      $planReqObj = @{ intent=$intent; paths=@("examples/sample_py"); code_paste=$snippet }
      $plan = Invoke-JsonPost -Uri ("http://127.0.0.1:" + $port + "/plan") -Object $planReqObj -TimeoutSec 90
      $planObj = $null
      if ($plan.plan -is [string]) { $planObj = $plan.plan | ConvertFrom-Json } else { $planObj = $plan.plan }
      if (-not $planObj) { throw "plan 비어있음" }
      $hint = "Return ONLY the items of the edits array (each item is a JSON object). No code fences or markdown. Use only actions: insert_before, insert_after, replace_range, delete_range."
      $psBodyObj = @{ plan=$planObj; feedback=@{ hint=$hint; reason="strict JSON required" } }
      # patch_smart만 호출 (apply/test 없음) → 서버측 success_logs에 자동 적재
      $null = Invoke-JsonPost -Uri ("http://127.0.0.1:" + $port + "/patch_smart") -Object $psBodyObj -TimeoutSec 120
      Write-Host ("[OK] [" + $rep + "] " + $intent)
    } catch {
      Write-Host ("[SKIP] [" + $rep + "] " + $intent + " : " + $_.Exception.Message) -ForegroundColor Yellow
    }
  }
}


