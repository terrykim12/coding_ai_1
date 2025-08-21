# UTF-8 인코딩 문제 해결된 API 테스트
param(
    [int]$Port = 8765,
    [string]$Path = "examples\sample_py\app.py",
    [string]$FunctionName = "add"
)

# UTF-8 설정
. .\tools\utf8.ps1

function Test-PlanApi {
    param(
        [string]$Intent,
        [string[]]$Paths,
        [string]$CodePaste
    )
    
    $url = "http://127.0.0.1:$Port/plan"
    
    # PowerShell 객체를 JSON으로 변환
    $body = @{
        intent = $Intent
        paths = $Paths
        code_paste = $CodePaste
    }
    
    try {
        # 방법 1: ConvertTo-Json 사용 (UTF-8 강제)
        $jsonBody = $body | ConvertTo-Json -Depth 10 -Compress
        
        Write-Output "요청 본문 길이: $($jsonBody.Length) 문자"
        
        # 방법 2: UTF-8 바이트로 변환 후 다시 문자열로
        $utf8Bytes = [System.Text.Encoding]::UTF8.GetBytes($jsonBody)
        $utf8String = [System.Text.Encoding]::UTF8.GetString($utf8Bytes)
        
        Write-Output "UTF-8 변환 후 길이: $($utf8String.Length) 문자"
        
        # API 호출
        $response = Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json; charset=utf-8" -Body $utf8String -TimeoutSec 180
        
        Write-Output "PLAN API 성공!"
        return $response
        
    } catch {
        Write-Output "PLAN API 실패: $($_.Exception.Message)"
        if ($_.Exception.Response) {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $responseBody = $reader.ReadToEnd()
            Write-Output "응답 본문: $responseBody"
        }
        return $null
    }
}

function Test-PatchApi {
    param(
        $Plan
    )
    
    $url = "http://127.0.0.1:$Port/patch"
    
    $body = @{
        plan = $Plan
    }
    
    try {
        $jsonBody = $body | ConvertTo-Json -Depth 100 -Compress
        $utf8Bytes = [System.Text.Encoding]::UTF8.GetBytes($jsonBody)
        $utf8String = [System.Text.Encoding]::UTF8.GetString($utf8Bytes)
        
        Write-Output "PATCH API 호출 중..."
        $response = Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json; charset=utf-8" -Body $utf8String -TimeoutSec 180
        
        Write-Output "PATCH API 성공!"
        return $response
        
    } catch {
        Write-Output "PATCH API 실패: $($_.Exception.Message)"
        return $null
    }
}

# 메인 실행
try {
    Write-Output "=== PowerShell API 테스트 시작 ==="
    
    # 파일에서 함수 스니펫 추출
    $src = Get-Content -Raw $Path
    $pat = '(?smi)^[ \t]*def[ \t]+' + [regex]::Escape($FunctionName) + '\(.*?\):\s*\r?\n(?:[ \t]+.*\r?\n)*'
    $m = [regex]::Match($src, $pat)
    $snippet = if ($m.Success) { $m.Value } else { $src }
    
    Write-Output "스니펫 길이: $($snippet.Length) 문자"
    
    # PLAN API 테스트
    $planResult = Test-PlanApi -Intent "add() 함수에 음수 방지 추가" -Paths @($Path) -CodePaste $snippet
    
    if ($planResult) {
        Write-Output "PLAN 생성 완료: $($planResult.plan_id)"
        
        # PATCH API 테스트
        $patchResult = Test-PatchApi -Plan $planResult.plan
        
        if ($patchResult) {
            Write-Output "PATCH 생성 완료: $($patchResult.patch_id)"
            Write-Output "=== 전체 워크플로우 성공 ==="
        } else {
            Write-Output "PATCH 생성 실패"
        }
    } else {
        Write-Output "PLAN 생성 실패"
    }
    
} catch {
    Write-Output "스크립트 실행 오류: $($_.Exception.Message)"
}
