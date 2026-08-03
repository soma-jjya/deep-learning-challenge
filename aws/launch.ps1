# AWS GPU 인스턴스 실행 스크립트
# 사용법: powershell -File aws\launch.ps1 [-Spot] [-Type g5.xlarge] [-Region ap-northeast-2]
param(
    [string]$Type = "g5.xlarge",
    [string]$Region = "ap-northeast-2",
    [switch]$Spot,
    [int]$DiskGB = 200
)
$ErrorActionPreference = "Stop"
$KeyName = "ajudl-gpu"
$SgName = "ajudl-gpu-sg"
$PemPath = "$env:USERPROFILE\.ssh\$KeyName.pem"

# 0. 자격 증명 확인
aws sts get-caller-identity --region $Region | Out-Null
if (-not $?) { Write-Host "aws configure 먼저 실행하세요"; exit 1 }

# 1. 최신 딥러닝 AMI (Ubuntu 22.04, NVIDIA 드라이버+PyTorch 포함)
$ami = aws ec2 describe-images --region $Region --owners amazon `
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" "Name=state,Values=available" `
    --query "sort_by(Images,&CreationDate)[-1].[ImageId,Name]" --output text
$amiId, $amiName = $ami -split "`t"
Write-Host "AMI: $amiId ($amiName)"

# 2. 키페어 (없으면 생성해 ~/.ssh에 저장)
$null = aws ec2 describe-key-pairs --key-names $KeyName --region $Region 2>$null
if (-not $?) {
    if (-not (Test-Path "$env:USERPROFILE\.ssh")) { New-Item -ItemType Directory "$env:USERPROFILE\.ssh" | Out-Null }
    aws ec2 create-key-pair --key-name $KeyName --region $Region --query KeyMaterial --output text | Out-File -Encoding ascii $PemPath
    Write-Host "키 생성: $PemPath"
}

# 3. 보안그룹 (내 IP만 SSH 허용)
$myIp = (Invoke-RestMethod "https://api.ipify.org")
$sgId = aws ec2 describe-security-groups --region $Region --filters "Name=group-name,Values=$SgName" --query "SecurityGroups[0].GroupId" --output text
if ($sgId -eq "None" -or [string]::IsNullOrEmpty($sgId)) {
    $sgId = aws ec2 create-security-group --group-name $SgName --description "ajudl gpu ssh" --region $Region --query GroupId --output text
    aws ec2 authorize-security-group-ingress --group-id $sgId --protocol tcp --port 22 --cidr "$myIp/32" --region $Region | Out-Null
    Write-Host "보안그룹 생성: $sgId (SSH from $myIp)"
} else {
    # IP가 바뀌었을 수 있으니 현재 IP 규칙 추가 시도 (이미 있으면 에러 무시)
    aws ec2 authorize-security-group-ingress --group-id $sgId --protocol tcp --port 22 --cidr "$myIp/32" --region $Region 2>$null | Out-Null
}

# 4. 인스턴스 시작
$marketOpts = @()
if ($Spot) { $marketOpts = @("--instance-market-options", "MarketType=spot,SpotOptions={SpotInstanceType=one-time}") }
$bootstrapPath = Join-Path $PSScriptRoot "bootstrap.sh"
$instanceId = aws ec2 run-instances --region $Region `
    --image-id $amiId --instance-type $Type `
    --key-name $KeyName --security-group-ids $sgId `
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DiskGB,VolumeType=gp3}" `
    --user-data "file://$bootstrapPath" `
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ajudl-train}]" `
    @marketOpts `
    --query "Instances[0].InstanceId" --output text
Write-Host "인스턴스 시작: $instanceId (type=$Type, spot=$($Spot.IsPresent))"
$instanceId | Out-File -Encoding ascii (Join-Path $PSScriptRoot "instance.txt")

# 5. 실행 대기 후 접속 정보 출력
aws ec2 wait instance-running --instance-ids $instanceId --region $Region
$ip = aws ec2 describe-instances --instance-ids $instanceId --region $Region `
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text
Write-Host ""
Write-Host "=== 준비 완료 (부팅 스크립트가 2~3분 더 돕니다) ==="
Write-Host "접속:   ssh -i $PemPath ubuntu@$ip"
Write-Host "확인:   ls ~/BOOTSTRAP_DONE 이 보이면 부팅 세팅 완료"
Write-Host "다음:   tmux new -s train  →  bash ~/setup_env.sh"
Write-Host ""
Write-Host "정지:   aws ec2 stop-instances --instance-ids $instanceId --region $Region"
Write-Host "삭제:   aws ec2 terminate-instances --instance-ids $instanceId --region $Region"
