# AWS 자율 실험 서버 — 격리·자동정지·무인 실행

목표: **기존 AWS 자원 불가침(최우선)** · 서버가 스스로 실험 실행·기록 · 유휴 시 자동 정지.

```
[로컬 Claude] ─ experiments/queue.md에 실험 정의 → git push
[EC2 g5.xlarge] ─ run_experiments.sh가 큐를 순서대로 실행
                  ├─ Claude Code(헤드리스)가 실행·분석·EXPERIMENTS.md 기록 → git push
                  ├─ ntfy로 폰 알림 (시작/완료/실패)
                  └─ 큐 비고 30분 유휴 → watchdog이 자동 stop (과금 정지)
[사용자] ─ 폰 알림 받고 GitHub(또는 로컬 git pull)에서 결과 확인만
```

## 기존 자원 불가침 — 3중 안전장치

1. **전용 IAM 사용자 + 태그 조건 정책** (`iam-policy.json`): `Project=ajudl` 태그가 붙은 자원만
   정지/삭제/변경 가능. 기존 자원은 **권한 차원에서** 건드릴 수 없음
2. **전용 VPC**: 기존 네트워크(기본 VPC 포함)를 쓰지 않고 `ajudl-vpc`(10.99.0.0/16)를 새로 만들어 그 안에서만 동작
3. **상태 파일** (`state.json`): 스크립트는 자기가 만든 자원 id만 기록·조작. teardown도 이 목록만 삭제

## 0. 최초 1회: 전용 IAM 사용자 만들기 (브라우저, 5분)

1. AWS 콘솔 → IAM → 정책 → 정책 생성 → JSON 탭 → `aws/iam-policy.json` 내용 붙여넣기 → 이름 `ajudl-experiment-policy`
2. IAM → 사용자 → 사용자 생성 → 이름 `ajudl-experiment` → 직접 정책 연결 → 위 정책 선택
3. 생성된 사용자 → 보안 자격 증명 → **액세스 키 만들기** (CLI 용도)
4. 로컬에서 **전용 프로필로** 등록 (기본 프로필을 오염시키지 않음):
   ```powershell
   aws configure --profile ajudl
   # 위 액세스 키 / region: ap-northeast-2 / output: json
   ```
5. GPU 쿼터 확인 (0이면 콘솔 Service Quotas에서 "Running On-Demand G and VT instances" 4로 증가 요청):
   ```powershell
   aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region ap-northeast-2 --profile ajudl
   ```

## 1. 인스턴스 시작 (로컬)

```powershell
powershell -File aws\launch.ps1          # 온디맨드 g5.xlarge (권장)
powershell -File aws\launch.ps1 -Spot    # 스팟 (중단 시 stop 후 수동 재시작)
```
전용 VPC·키·보안그룹(내 IP만 SSH) 생성 → 인스턴스 시작 → SSH 명령 출력.
비용: g5.xlarge 온디맨드 ~$1.3/h. **유휴 30분이면 자동 stop**되므로 상시대기 과금 없음.

## 2. 서버 최초 1회 세팅 (SSH)

필요 토큰 4개: GitHub PAT(레포 private) · Claude 구독 토큰(`claude setup-token`, 노트북에서) ·
HF 토큰 · wandb 토큰. ntfy 주제는 아무 비밀 문자열이나 정하면 됨.

```bash
ssh -i ~/.ssh/ajudl-gpu.pem ubuntu@<IP>
echo 'export GITHUB_TOKEN="github_pat_..."'          >> ~/.bashrc
echo 'export CLAUDE_CODE_OAUTH_TOKEN="sk-ant-..."'   >> ~/.bashrc
echo 'export NTFY_TOPIC="ajudl-비밀문자열"'           >> ~/.bashrc
source ~/.bashrc
tmux new -s train
bash ~/setup_env.sh          # 레포 클론 + 파이썬 환경 + watchdog cron 등록
huggingface-cli login && wandb login
```

폰 알림 구독: 폰 브라우저(또는 ntfy 앱)에서 `ntfy.sh/ajudl-비밀문자열` 열어두기.

## 3. 무인 실험 루프

```bash
# 서버 tmux 안에서 러너 시작 — 이후는 전부 자동
cd ~/work/deep-learning-challenge && git pull
nohup bash remote/run_experiments.sh > runner.log 2>&1 &
```
- 러너가 `experiments/queue.md`의 `- [ ]` 항목을 위에서부터 Claude에게 실행시킴
- 실험마다: 결과를 EXPERIMENTS.md에 기록 → 큐 체크 → git push → 폰 알림
- 큐가 비면 러너 종료 → 30분 뒤 watchdog이 인스턴스 stop
- **새 실험 추가**: 로컬에서 queue.md에 줄 추가 → push → `aws\start.ps1` → 서버에서 러너 재시작
- 수동 작업할 땐 자동정지 잠금: `touch ~/KEEP_ALIVE` (해제: `rm ~/KEEP_ALIVE`)

## 4. 일상 명령 (로컬)

```powershell
powershell -File aws\start.ps1      # 재시작 + 새 IP 출력
powershell -File aws\stop.ps1       # 수동 정지
powershell -File aws\teardown.ps1   # 프로젝트 자원 전부 삭제 (state.json 기록분만, 확인 후)
```

## 결과 받아보는 법 (사용자)

1. 폰 알림: 실험 시작/완료/실패가 ntfy로 옴
2. GitHub에서 EXPERIMENTS.md / results/ 확인, 또는 로컬 Claude 세션에서 "결과 분석해줘" (git pull 후 report.html 갱신까지)
