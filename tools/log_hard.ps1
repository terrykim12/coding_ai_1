param([int]$N=200, [string]$ServerHost="127.0.0.1", [int]$Port=8765)
$funcs = @("add","divide","fibonacci","is_prime","factorial")
$intents = @(
 "add()에 타입/음수 입력 방지와 예외처리 추가",
 "divide() 0 나누기 방지와 오류 메시지 개선",
 "fibonacci() n==0/1 경계 처리 및 입력검증",
 "is_prime() 에라토스테네스 최적화 힌트",
 "factorial() 재귀→반복 변환 및 큰수 보호"
)
function Get-RandSnippet([string]$p,[string]$f){
  $raw = Get-Content -Raw -Encoding UTF8 $p
  $len = [Math]::Min($raw.Length, (Get-Random -Minimum 600 -Maximum 1200))
  return $raw.Substring(0,$len)
}
for($i=1;$i -le $N;$i++){
  $f = $funcs | Get-Random
  $intent = $intents | Get-Random
  $src = Get-RandSnippet "examples\sample_py\app.py" $f
  $planBody = @{ intent=$intent; paths=@(); code_paste=$src } | ConvertTo-Json -Depth 40 -Compress
  try{
    $plan = Invoke-RestMethod "http://${ServerHost}:${Port}/plan" -Method Post -ContentType 'application/json; charset=utf-8' -Body $planBody -TimeoutSec 30
    $planObj = if ($plan.plan -is [string]) { $plan.plan | ConvertFrom-Json } else { $plan.plan }
    $fb = @{ hint="Return ONLY items of the edits array. No markdown."; reason="log" }
    $patchBody = @{ plan=$planObj; feedback=$fb } | ConvertTo-Json -Depth 100
    $null = Invoke-RestMethod "http://${ServerHost}:${Port}/patch_smart" -Method Post -ContentType 'application/json; charset=utf-8' -Body $patchBody -TimeoutSec 40
    Write-Host "[$i/$N] logged" -ForegroundColor Green
  }catch{ 
    Write-Host "[$i/$N] fail: $($_.Exception.Message)" -ForegroundColor Red 
  }
  Start-Sleep -Milliseconds 200
}
