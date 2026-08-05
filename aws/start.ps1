# 정지된 인스턴스 재시작 + 새 IP 출력 (+ 필요 시 내 IP를 SSH 허용에 추가)
$ErrorActionPreference = "Stop"
$Profile = "ajudl"; $Region = "ap-northeast-2"
$state = Get-Content (Join-Path $PSScriptRoot "state.json") | ConvertFrom-Json

aws ec2 start-instances --instance-ids $state.instanceId --profile $Profile --region $Region | Out-Null
aws ec2 wait instance-running --instance-ids $state.instanceId --profile $Profile --region $Region

$myIp = (Invoke-RestMethod "https://api.ipify.org")
cmd /c "aws ec2 authorize-security-group-ingress --group-id $($state.sgId) --protocol tcp --port 22 --cidr $myIp/32 --profile $Profile --region $Region >nul 2>&1"

$ip = aws ec2 describe-instances --instance-ids $state.instanceId --profile $Profile --region $Region `
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text
Write-Host "실행 중. 접속:  ssh -i $env:USERPROFILE\.ssh\ajudl-gpu.pem ubuntu@$ip"
Write-Host "러너 시작:      tmux new -s train → cd ~/work/deep-learning-challenge && git pull && nohup bash remote/run_experiments.sh > runner.log 2>&1 &"
