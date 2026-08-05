# 인스턴스 수동 정지 (GPU 과금 정지, 디스크 유지)
$ErrorActionPreference = "Stop"
$state = Get-Content (Join-Path $PSScriptRoot "state.json") | ConvertFrom-Json
aws ec2 stop-instances --instance-ids $state.instanceId --profile ajudl --region ap-northeast-2 | Out-Null
Write-Host "정지 요청됨: $($state.instanceId)"
