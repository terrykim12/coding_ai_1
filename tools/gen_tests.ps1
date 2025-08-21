param(
	[Parameter(Mandatory=$true)][string]$Path,
	[Parameter(Mandatory=$true)][string]$Function,
	[string]$Port="8765"
)
$ErrorActionPreference="Stop"

$src = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
$prompt = @"
You are a Python unit test generator.
Given the function below, write a minimal pytest test file named test_$Function.py that:
- imports the function
- covers at least 3 cases including an edge case
- asserts expected outputs
- no external I/O

<CODE>
$src
</CODE>
Return only the test code.
"@

$body = @{ prompt=$prompt; stream=$false; options=@{ num_predict=256; temperature=0.0 } } | ConvertTo-Json -Depth 10
$res = Invoke-RestMethod "http://127.0.0.1:$Port/api/generate" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 40
$code = $res.response
$testFile = Join-Path (Split-Path $Path -Parent) "test_$Function.py"
$code | Out-File -Encoding UTF8 -LiteralPath $testFile
Write-Host "Wrote $testFile" -ForegroundColor Green
