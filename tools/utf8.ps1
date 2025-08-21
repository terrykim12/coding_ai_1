Set-StrictMode -Version Latest
try { chcp 65001 | Out-Null } catch {}
$enc = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding  = $enc
[Console]::OutputEncoding = $enc
$script:OutputEncoding    = $enc
$PSDefaultParameterValues['Out-File:Encoding']   = 'utf8'
$PSDefaultParameterValues['Set-Content:Encoding']= 'utf8'
$PSDefaultParameterValues['Add-Content:Encoding']= 'utf8'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
Write-Output 'UTF-8 설정 완료'

