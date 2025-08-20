# test_api.ps1
#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

# 한글 입출력 깨짐 방지
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$base = 'http://127.0.0.1:8765'
$headers = @{ 'Content-Type' = 'application/json; charset=utf-8' }

# 1) /plan
$planBody = @{
    intent     = 'add() 함수에 음수 검증 추가'
    paths      = @('examples/sample_py')
    code_paste = 'def add(a,b): return a+b'
} | ConvertTo-Json -Depth 10 -Compress

$plan = Invoke-RestMethod -Method Post -Uri "$base/plan" -Headers $headers -Body $planBody

# 2) /patch
$patchBody = @{ plan = $plan.plan } | ConvertTo-Json -Depth 20 -Compress
$patch = Invoke-RestMethod -Method Post -Uri "$base/patch" -Headers $headers -Body $patchBody

# 3) /apply
$applyBody = @{ patch = $patch.patch } | ConvertTo-Json -Depth 20 -Compress
$apply = Invoke-RestMethod -Method Post -Uri "$base/apply" -Headers $headers -Body $applyBody

# 4) /test
$test = Invoke-RestMethod -Method Post -Uri "$base/test" -Headers $headers -Body '{}'

# 결과 확인
'--- PLAN ---'
$plan | ConvertTo-Json -Depth 10
'--- PATCH ---'
$patch | ConvertTo-Json -Depth 10
'--- APPLY ---'
$apply | ConvertTo-Json -Depth 10
'--- TEST ---'
$test | ConvertTo-Json -Depth 10

# 한 줄로 빠르게 시험해보기 (PowerShell 표준 방식)
# $body = '{"intent":"add() 함수에 음수 검증 추가","paths":["examples/sample_py"],"code_paste":"def add(a,b): return a+b"}'
# Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/plan -ContentType 'application/json; charset=utf-8' -Body $body

# 리눅스식 curl 플래그를 꼭 쓰고 싶다면
# PowerShell에서 진짜 curl을 호출하세요: curl.exe
# $json = @'
# {"intent":"add() 함수에 음수 검증 추가","paths":["examples/sample_py"],"code_paste":"def add(a,b): return a+b"}
# '@
# curl.exe -s -X POST "http://127.0.0.1:8765/plan" -H "Content-Type: application/json; charset=utf-8" -d $json

# 인코딩 저장 방법 (VS Code)
# 우하단 상태바 "UTF-8" 클릭 → "Save with Encoding" → "UTF-8 with BOM" 선택 → 저장.

# 참고: 포트 충돌 체크(선택)
# 가끔 8765 포트가 이미 점유되면 호출이 실패합니다.
# netstat -ano | findstr :8765
# PID가 보이면:
# taskkill /PID <PID> /F

