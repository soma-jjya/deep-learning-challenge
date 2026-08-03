# Kaggle 노트북을 원격 실행하고 완료까지 감시한 뒤 로그·결과를 받아온다.
# 사용법: powershell -File scripts\kaggle_run.ps1 [-PollSec 300]
param([int]$PollSec = 300)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$meta = Get-Content (Join-Path $repo "kaggle\kernel-metadata.json") | ConvertFrom-Json
$slug = $meta.id

Write-Host "푸시: $slug ($($meta.code_file))"
python -m kaggle kernels push -p (Join-Path $repo "kaggle")
if (-not $?) { exit 1 }

Write-Host "실행 시작됨. $PollSec 초 간격으로 상태 확인..."
while ($true) {
    Start-Sleep -Seconds $PollSec
    $status = python -m kaggle kernels status $slug 2>&1 | Out-String
    $time = Get-Date -Format "HH:mm"
    Write-Host "[$time] $($status.Trim())"
    if ($status -match "complete") { break }
    if ($status -match "error|cancel") { Write-Host "실행 실패 — 로그를 받아 확인합니다"; break }
}

$outDir = Join-Path $repo ("runs\" + (Get-Date -Format "yyyyMMdd_HHmm"))
New-Item -ItemType Directory -Force $outDir | Out-Null
python -m kaggle kernels output $slug -p $outDir
Write-Host ""
Write-Host "=== 결과 요약 (실험 점수 라인) ==="
Get-ChildItem $outDir -Recurse -Include *.log,*.txt | ForEach-Object {
    Select-String -Path $_ -Pattern "exp0|정확도|Accuracy|Error|Traceback" | Select-Object -First 20
}
Write-Host "전체 출력: $outDir"
