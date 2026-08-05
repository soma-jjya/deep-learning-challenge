# AWS GPU 인스턴스 실행 — 완전 격리 버전
# 원칙: ① 전용 프로필(ajudl)만 사용 ② 모든 자원은 전용 VPC에 새로 생성 + Project=ajudl 태그
#       ③ 이 스크립트가 만든 자원 id는 aws/state.json에 기록, 그 외에는 절대 조작하지 않음
# 사용법: powershell -File aws\launch.ps1 [-Spot] [-Type g5.xlarge]
param(
    [string]$Type = "g5.xlarge",
    [string]$Region = "ap-northeast-2",
    [switch]$Spot,
    [int]$DiskGB = 200
)
$ErrorActionPreference = "Stop"
$Profile = "ajudl"
$KeyName = "ajudl-gpu"
$PemPath = "$env:USERPROFILE\.ssh\$KeyName.pem"
$StatePath = Join-Path $PSScriptRoot "state.json"
$Tag = "ResourceType=REPLACE,Tags=[{Key=Project,Value=ajudl},{Key=Name,Value=REPLACE_NAME}]"

function AwsCmd { param([string[]]$CmdArgs)
    $out = aws @CmdArgs --profile $Profile --region $Region --output text
    if (-not $?) { throw "aws 명령 실패: $($CmdArgs -join ' ')" }
    return $out
}

# 0. 전용 프로필 확인 — 기본 프로필은 절대 쓰지 않는다
aws sts get-caller-identity --profile $Profile --region $Region | Out-Null
if (-not $?) {
    Write-Host "전용 프로필이 없습니다. 먼저 실행:  aws configure --profile ajudl"
    Write-Host "(ajudl 전용 IAM 사용자의 액세스 키 사용 — aws/README.md 0장 참고)"
    exit 1
}

# 상태 파일 로드
$state = @{}
if (Test-Path $StatePath) {
    (Get-Content $StatePath | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $state[$_.Name] = $_.Value }
}

# 1. 전용 VPC (기존 네트워크 불가침 — 우리 것만 새로 만든다)
if (-not $state.vpcId) {
    $vpcId = AwsCmd @("ec2","create-vpc","--cidr-block","10.99.0.0/16",
        "--tag-specifications","ResourceType=vpc,Tags=[{Key=Project,Value=ajudl},{Key=Name,Value=ajudl-vpc}]",
        "--query","Vpc.VpcId")
    AwsCmd @("ec2","modify-vpc-attribute","--vpc-id",$vpcId,"--enable-dns-hostnames") | Out-Null
    $subnetId = AwsCmd @("ec2","create-subnet","--vpc-id",$vpcId,"--cidr-block","10.99.1.0/24",
        "--tag-specifications","ResourceType=subnet,Tags=[{Key=Project,Value=ajudl},{Key=Name,Value=ajudl-subnet}]",
        "--query","Subnet.SubnetId")
    AwsCmd @("ec2","modify-subnet-attribute","--subnet-id",$subnetId,"--map-public-ip-on-launch") | Out-Null
    $igwId = AwsCmd @("ec2","create-internet-gateway",
        "--tag-specifications","ResourceType=internet-gateway,Tags=[{Key=Project,Value=ajudl},{Key=Name,Value=ajudl-igw}]",
        "--query","InternetGateway.InternetGatewayId")
    AwsCmd @("ec2","attach-internet-gateway","--internet-gateway-id",$igwId,"--vpc-id",$vpcId) | Out-Null
    $rtbId = AwsCmd @("ec2","describe-route-tables","--filters","Name=vpc-id,Values=$vpcId","--query","RouteTables[0].RouteTableId")
    AwsCmd @("ec2","create-tags","--resources",$rtbId,"--tags","Key=Project,Value=ajudl","Key=Name,Value=ajudl-rtb") | Out-Null
    AwsCmd @("ec2","create-route","--route-table-id",$rtbId,"--destination-cidr-block","0.0.0.0/0","--gateway-id",$igwId) | Out-Null
    $state.vpcId = $vpcId; $state.subnetId = $subnetId; $state.igwId = $igwId; $state.rtbId = $rtbId
    $state | ConvertTo-Json | Out-File -Encoding utf8 $StatePath   # 즉시 기록 (고아 자원 방지)
    Write-Host "전용 VPC 생성: $vpcId"
}

# 2. 키페어 (cmd /c로 실행해 stderr가 예외로 승격되는 PS5.1 문제 회피)
cmd /c "aws ec2 describe-key-pairs --key-names $KeyName --profile $Profile --region $Region >nul 2>&1"
if ($LASTEXITCODE -ne 0) {
    if (-not (Test-Path "$env:USERPROFILE\.ssh")) { New-Item -ItemType Directory "$env:USERPROFILE\.ssh" | Out-Null }
    aws ec2 create-key-pair --key-name $KeyName --profile $Profile --region $Region `
        --tag-specifications "ResourceType=key-pair,Tags=[{Key=Project,Value=ajudl}]" `
        --query KeyMaterial --output text | Out-File -Encoding ascii $PemPath
    Write-Host "키 생성: $PemPath"
}
$state.keyName = $KeyName

# 3. 보안그룹 (전용 VPC 안, 내 IP만 SSH)
$myIp = (Invoke-RestMethod "https://api.ipify.org")
if (-not $state.sgId) {
    $sgId = AwsCmd @("ec2","create-security-group","--group-name","ajudl-gpu-sg",
        "--description","ajudl gpu ssh","--vpc-id",$state.vpcId,
        "--tag-specifications","ResourceType=security-group,Tags=[{Key=Project,Value=ajudl}]",
        "--query","GroupId")
    $state.sgId = $sgId
    $state | ConvertTo-Json | Out-File -Encoding utf8 $StatePath
}
cmd /c "aws ec2 authorize-security-group-ingress --group-id $($state.sgId) --protocol tcp --port 22 --cidr $myIp/32 --profile $Profile --region $Region >nul 2>&1"

# 4. user-data 준비 — CRLF→LF 정규화 + BOM 없는 UTF-8로 임시 파일 생성, fileb://로 전달
$raw = [IO.File]::ReadAllText((Join-Path $PSScriptRoot "bootstrap.sh"))
$tmpUd = Join-Path $env:TEMP "ajudl_userdata.sh"
[IO.File]::WriteAllText($tmpUd, ($raw -replace "`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))

# 5. AMI + 인스턴스
$amiId = AwsCmd @("ec2","describe-images","--owners","amazon",
    "--filters","Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*","Name=state,Values=available",
    "--query","sort_by(Images,&CreationDate)[-1].ImageId")
Write-Host "AMI: $amiId"

$runArgs = @("ec2","run-instances",
    "--image-id",$amiId,"--instance-type",$Type,
    "--key-name",$KeyName,"--security-group-ids",$state.sgId,"--subnet-id",$state.subnetId,
    "--block-device-mappings","DeviceName=/dev/sda1,Ebs={VolumeSize=$DiskGB,VolumeType=gp3}",
    "--user-data","fileb://$tmpUd",
    "--instance-initiated-shutdown-behavior","stop",
    "--tag-specifications","ResourceType=instance,Tags=[{Key=Project,Value=ajudl},{Key=Name,Value=ajudl-train}]",
    "--query","Instances[0].InstanceId")
if ($Spot) { $runArgs += @("--instance-market-options","MarketType=spot,SpotOptions={SpotInstanceType=persistent,InstanceInterruptionBehavior=stop}") }
$instanceId = AwsCmd $runArgs
$state.instanceId = $instanceId
$state | ConvertTo-Json | Out-File -Encoding utf8 $StatePath
Write-Host "인스턴스 시작: $instanceId (type=$Type, spot=$($Spot.IsPresent))"

aws ec2 wait instance-running --instance-ids $instanceId --profile $Profile --region $Region
$ip = AwsCmd @("ec2","describe-instances","--instance-ids",$instanceId,"--query","Reservations[0].Instances[0].PublicIpAddress")
Write-Host ""
Write-Host "=== 준비 완료 (부팅 스크립트 2~3분 더 소요) ==="
Write-Host "접속:  ssh -i $PemPath ubuntu@$ip"
Write-Host "다음:  tmux new -s train  →  bash ~/setup_env.sh  (aws/README.md 2장)"
