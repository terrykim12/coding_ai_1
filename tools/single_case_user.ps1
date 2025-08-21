# single_case_user.ps1 수정버전
param(
    [int]$Port = 8765,
    [string]$Path = "examples\sample_py\app.py",
    [string]$FunctionName = "add"
)

# UTF-8 인코딩 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

function Invoke-ApiRequest {
    param(
        [string]$Uri,
        [string]$Body
    )
    
    try {
        return Invoke-RestMethod -Method Post -Uri $Uri -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($Body))
    } catch {
        Write-Host "API 요청 실패: $($_.Exception.Message)" -ForegroundColor Red
        if ($_.ErrorDetails.Message) {
            Write-Host "서버 응답: $($_.ErrorDetails.Message)" -ForegroundColor Yellow
        }
        throw
    }
}

try {
    # 1. 소스 코드 읽기
    $src = Get-Content -Raw $Path -Encoding UTF8
    
    # 2. 함수 추출
    $pat = '(?smi)^[ \t]*def[ \t]+' + [regex]::Escape($FunctionName) + '\(.*?\):\s*\r?\n(?:[ \t]+.*\r?\n)*'
    $m = [regex]::Match($src, $pat)
    $snippet = if($m.Success) { $m.Value } else { $src }
    
    # 3. 플랜 요청 생성
    $planRequest = @{
        intent = "$FunctionName() 음수 방지"
        paths = @("examples\sample_py")
        code_paste = $snippet
    }
    
    $planRequestJson = $planRequest | ConvertTo-Json -Depth 10 -Compress
    Write-Host "플랜 요청 JSON: $planRequestJson" -ForegroundColor Cyan
    
    # 4. 플랜 생성 요청
    Write-Host "플랜 생성 중..." -ForegroundColor Green
    $planUri = "http://127.0.0.1:$Port/plan"
    
    $planResponse = Invoke-ApiRequest -Uri $planUri -Body $planRequestJson
    Write-Host "플랜 생성 성공!" -ForegroundColor Green
    Write-Host ($planResponse | ConvertTo-Json -Depth 5) -ForegroundColor White
    
    # 5. 패치 요청 생성
    $patchRequest = @{
        plan = $planResponse.plan
        feedback = @{
            hint = "Return ONLY the items of the edits array (each item is JSON). No fences/markdown."
            reason = "log only"
        }
    }
    
    $patchRequestJson = $patchRequest | ConvertTo-Json -Depth 20 -Compress
    
    # 6. 패치 생성 요청 (smart 먼저 시도)
    Write-Host "스마트 패치 생성 중..." -ForegroundColor Green
    $patchSmartUri = "http://127.0.0.1:$Port/patch_smart"
    
    try {
        $patchResponse = Invoke-ApiRequest -Uri $patchSmartUri -Body $patchRequestJson
        Write-Host "스마트 패치 생성 성공!" -ForegroundColor Green
        Write-Host ($patchResponse | ConvertTo-Json -Depth 5) -ForegroundColor White
    } catch {
        Write-Host "스마트 패치 실패, 일반 패치로 시도..." -ForegroundColor Yellow
        
        $patchUri = "http://127.0.0.1:$Port/patch"
        $patchResponse = Invoke-ApiRequest -Uri $patchUri -Body $patchRequestJson
        Write-Host "일반 패치 생성 성공!" -ForegroundColor Green
        Write-Host ($patchResponse | ConvertTo-Json -Depth 5) -ForegroundColor White
    }
    
} catch {
    Write-Host "전체 프로세스 실패: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}


