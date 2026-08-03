# AWS GPU 실험 서버 세팅 가이드

remote-finetune-session.md의 "항상 켜진 GPU 서버"를 AWS EC2로 만드는 절차.
목표 구조: **EC2(GPU) + tmux 상주 세션 + Claude Code 헤드리스** → 노트북을 꺼도 실험이 계속 돈다.

```
[내 노트북] ──ssh──▶ [EC2 g5.xlarge (A10G 24GB)]
                      └─ tmux ─┬─ 학습 (nohup, 로그 파일)
                               └─ claude -p "다음 실험..." (오케스트레이터)
```

## 비용 (서울 리전, 시간당 대략)

| 인스턴스 | GPU | 온디맨드 | 스팟 | 용도 |
|---|---|---|---|---|
| g4dn.xlarge | T4 16GB | ~$0.65 | ~$0.2 | 최저가. Kaggle과 동일 GPU |
| **g5.xlarge** ⭐ | A10G 24GB | ~$1.3 | ~$0.4~0.6 | QLoRA 여유 + bf16 지원(T4보다 ~2배 빠름) |

- **안 쓸 때 stop** (EBS 보관비 월 $8/100GB만 나감). terminate하면 디스크까지 삭제
- 스팟은 저렴하지만 중간 회수될 수 있음 → 체크포인트 저장 필수 (launch.ps1의 `-Spot` 옵션)

## 0. 최초 1회: 계정 준비 (브라우저에서 직접)

1. AWS 계정 생성 (있으면 스킵) → 결제 수단 등록
2. IAM에서 액세스 키 발급: 콘솔 → IAM → 사용자 → 본인 → 보안 자격 증명 → 액세스 키 만들기 (CLI 용도)
3. 로컬에서 자격 증명 등록:
   ```powershell
   aws configure
   # Access Key ID / Secret 입력, region: ap-northeast-2, output: json
   ```
4. **GPU 쿼터 확인 (신규 계정은 0이라 인스턴스를 못 띄움!)**
   ```powershell
   aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region ap-northeast-2
   # "Running On-Demand G and VT instances" — Value가 4 이상이어야 g5.xlarge 가능
   ```
   0이면 콘솔 → Service Quotas → EC2 → "Running On-Demand G and VT instances" → 증가 요청(4).
   승인까지 몇 시간~2일. (스팟 쓰려면 "All G and VT Spot Instance Requests"도 동일하게)

## 1. 인스턴스 띄우기 (로컬에서)

```powershell
cd C:\Users\82108\Desktop\harness_project\ajudl
# 온디맨드 (기본, 안정적)
powershell -File aws\launch.ps1
# 스팟 (저렴, 회수 가능성 있음)
powershell -File aws\launch.ps1 -Spot
```

스크립트가 하는 일: 최신 딥러닝 AMI 조회 → 키페어·보안그룹(내 IP만 SSH 허용) 생성 →
인스턴스 시작(디스크 200GB) → 부팅 시 aws/bootstrap.sh 자동 실행(tmux·Node·Claude Code 설치) →
접속용 SSH 명령 출력.

## 2. 서버 접속 후 최초 1회 세팅

```bash
ssh -i ~/.ssh/ajudl-gpu.pem ubuntu@<출력된 IP>
tmux new -s train                 # 상주 세션 (이후 재접속은 tmux attach -t train)
bash ~/setup_env.sh               # 파이썬 환경 + Unsloth + 레포 클론 (bootstrap이 복사해 둠)
```

Claude Code 구독 인증 (API 키 아님 — 과금 없음):
```powershell
# [노트북에서] claude setup-token   → 토큰 복사
```
```bash
# [서버에서]
echo 'export CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-..."' >> ~/.bashrc && source ~/.bashrc
claude --version && claude -p "hello" # 응답 오면 성공
```

HF·wandb 로그인 (학습 로그 = 검증 제출물 권장 항목):
```bash
huggingface-cli login    # HF 토큰
wandb login              # wandb 토큰
```

## 3. 실험 루프 (반복)

```bash
tmux attach -t train
cd ~/work/deep-learning-challenge
git pull                                          # 로컬에서 푸시한 최신 실험 코드 받기
nohup uv run python remote/train_qlora.py > train.log 2>&1 &   # 학습은 백그라운드
tail -f train.log                                 # 지켜보다 Ctrl+C (학습은 계속 돎)
# 학습 끝나면:
claude -p "train.log와 eval 결과를 읽고 EXPERIMENTS.md 형식으로 요약해줘"
```

노트북을 꺼도 학습은 계속된다. 결과 확인은 재접속 → `tmux attach -t train`.

## 4. 끝나면 반드시

```powershell
# stop: 디스크 유지, GPU 과금 정지 (다음에 start로 재개, IP는 바뀜)
aws ec2 stop-instances --instance-ids <id> --region ap-northeast-2
# terminate: 완전 삭제 (체크포인트를 HF Hub에 백업했는지 확인 후!)
aws ec2 terminate-instances --instance-ids <id> --region ap-northeast-2
```

launch.ps1이 인스턴스 id를 `aws/instance.txt`에 기록해 둔다.
