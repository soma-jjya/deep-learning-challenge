# 현재 공인 IP를 보안그룹의 SSH 허용에 등록한다 (멱등, 이미 있으면 조용히 통과).
#
# 왜 필요한가: 이 프로젝트에서 "SSH가 간헐적으로 막힌다"고 반복 기록해온 현상의
# 실제 원인은 네트워크 차단이 아니라 **ISP의 공인 IP 로테이션**이었다(2026-08-11 확인:
# 한 세션 안에서 211.235.66.89 -> 58.231.140.230으로 변경). IP가 바뀌면 보안그룹
# 규칙에서 이탈해 접속이 timeout 난다. SSH가 안 되면 먼저 이 스크립트를 돌릴 것.
#
# 사용: powershell -File aws\ensure-ssh.ps1
# 참고: 오래된 IP 규칙이 쌓이면 -Prune 으로 정리 (보안그룹 규칙 수 상한 대비)

param([switch]$Prune)

$ErrorActionPreference = "Stop"
$AwsProfile = "ajudl"; $Region = "ap-northeast-2"
$state = Get-Content (Join-Path $PSScriptRoot "state.json") | ConvertFrom-Json

$myIp = (Invoke-RestMethod "https://api.ipify.org").Trim()
Write-Host "현재 공인 IP: $myIp"

# stderr가 예외로 승격되는 것을 막기 위해 cmd /c 경유 (PowerShell 5.1 이슈)
cmd /c "aws ec2 authorize-security-group-ingress --group-id $($state.sgId) --protocol tcp --port 22 --cidr $myIp/32 --profile $AwsProfile --region $Region >nul 2>&1"

if ($Prune) {
    $rulesJson = cmd /c "aws ec2 describe-security-group-rules --filters Name=group-id,Values=$($state.sgId) --profile $AwsProfile --region $Region --output json 2>nul"
    $rules = ($rulesJson | ConvertFrom-Json).SecurityGroupRules
    $stale = $rules | Where-Object {
        -not $_.IsEgress -and $_.FromPort -eq 22 -and $_.CidrIpv4 -and $_.CidrIpv4 -ne "$myIp/32"
    }
    foreach ($r in $stale) {
        Write-Host "  오래된 규칙 제거: $($r.CidrIpv4)"
        cmd /c "aws ec2 revoke-security-group-ingress --group-id $($state.sgId) --security-group-rule-ids $($r.SecurityGroupRuleId) --profile $AwsProfile --region $Region >nul 2>&1"
    }
}

$ip = cmd /c "aws ec2 describe-instances --instance-ids $($state.instanceId) --profile $AwsProfile --region $Region --query ""Reservations[0].Instances[0].[State.Name,PublicIpAddress]"" --output text 2>nul"
Write-Host "인스턴스: $ip"
Write-Host "접속:  ssh -i `$env:USERPROFILE\.ssh\ajudl-gpu.pem ubuntu@$(($ip -split '\s+')[1])"
