# 이 프로젝트가 만든 자원'만' 완전 삭제 — state.json에 기록된 id만 조작한다.
# (IAM 정책상 Project=ajudl 태그가 없는 자원은 어차피 삭제 권한이 없음)
$ErrorActionPreference = "Stop"
$Profile = "ajudl"; $Region = "ap-northeast-2"
$StatePath = Join-Path $PSScriptRoot "state.json"
$state = Get-Content $StatePath | ConvertFrom-Json

Write-Host "삭제 대상 (state.json 기록분만):"
$state | Format-List
$confirm = Read-Host "체크포인트 백업을 확인했습니까? 전부 삭제하려면 'delete' 입력"
if ($confirm -ne "delete") { Write-Host "취소됨"; exit 0 }

if ($state.instanceId) {
    aws ec2 terminate-instances --instance-ids $state.instanceId --profile $Profile --region $Region | Out-Null
    aws ec2 wait instance-terminated --instance-ids $state.instanceId --profile $Profile --region $Region
    Write-Host "인스턴스 삭제됨"
}
if ($state.sgId)    { aws ec2 delete-security-group --group-id $state.sgId --profile $Profile --region $Region }
if ($state.igwId)   {
    aws ec2 detach-internet-gateway --internet-gateway-id $state.igwId --vpc-id $state.vpcId --profile $Profile --region $Region
    aws ec2 delete-internet-gateway --internet-gateway-id $state.igwId --profile $Profile --region $Region
}
if ($state.subnetId){ aws ec2 delete-subnet --subnet-id $state.subnetId --profile $Profile --region $Region }
if ($state.vpcId)   { aws ec2 delete-vpc --vpc-id $state.vpcId --profile $Profile --region $Region }
if ($state.keyName) { aws ec2 delete-key-pair --key-name $state.keyName --profile $Profile --region $Region }
Remove-Item $StatePath
Write-Host "전용 자원 전부 삭제 완료"
