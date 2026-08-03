# 기존 GPU 서버에서 Claude Code로 Qwen 3B 수학 튜닝 자동 세션

이미 있는 GPU 서버(직접 보유 / RunPod·Vast 대여 / 학교·회사 서버)에서
Claude Code(구독형)로 **Qwen 3B를 수학 특화 모델로 QLoRA 튜닝**하고,
**노트북을 꺼도 서버에서 알아서 돌아가게** 만드는 가이드.

AWS·ComfyUI는 필요 없다. 핵심은 딱 두 가지다:
**① 서버가 안 꺼진다, ② tmux로 세션을 서버에 상주시킨다.**

```
[내 노트북]  ──ssh──▶  [항상 켜진 GPU 서버]
                         └─ tmux 세션 ─┬─ 학습 프로세스(파이썬, GPU에서 몇 시간~며칠)
 노트북 꺼도               (서버에 상주)  └─ Claude Code (실험 오케스트레이터)
 이건 계속 돎
```

노트북을 끄면 SSH만 끊긴다. tmux 세션은 서버에 살아 있으므로 학습은 계속된다.
다음에 `ssh` + `tmux attach`로 다시 붙으면 그동안 돌아간 결과가 그대로 있다.

---

## 0. 서버가 갖춰야 할 조건

| 조건 | 확인 방법 / 기준 |
|---|---|
| **항상 켜져 있음** | 남이 안 끄는 서버. **본인 노트북/데스크톱은 안 됨**(꺼지니까). |
| SSH 접속 가능 | `ssh 사용자@서버주소` 로 붙을 수 있어야 함 |
| GPU VRAM ≥ 16GB | 아래 표. Qwen 3B QLoRA는 12GB로도 되지만 16GB↑ 권장 |
| sudo 또는 패키지 설치 권한 | Node·tmux·파이썬 라이브러리 설치용 |
| NVIDIA 드라이버 설치됨 | `nvidia-smi`로 GPU가 보이면 OK |

### VRAM별로 할 수 있는 것 (Qwen 3B 기준)
| VRAM | 가능한 방식 | 비고 |
|---|---|---|
| 12GB (예: RTX 3060/4070) | QLoRA(4bit) | 배치·시퀀스 길이를 작게. 가능은 함 |
| **16~24GB (T4·A10G·4090)** | **QLoRA 여유롭게, LoRA도** | 대회용으로 가장 무난 ⭐ |
| 40GB+ (A100·H100) | 풀 파인튜닝까지 | 3B 전체 파라미터 학습 가능 |

> QLoRA(4비트 양자화 + LoRA 어댑터)는 3B 모델을 **8~10GB** 정도로 학습한다.
> 그래서 16GB면 넉넉하고, 12GB도 설정을 조이면 된다. 풀 파인튜닝만 40GB+가 필요하다.

---

## 1. 서버 접속

RunPod·Vast 같은 대여 서비스는 대시보드에 SSH 명령이 그대로 나온다. 직접 서버면:
```bash
ssh 사용자@서버주소
nvidia-smi     # GPU와 VRAM 확인. 여기서 GPU가 보여야 시작 가능
```

> 대여 서비스는 **인스턴스를 stop/terminate하면 디스크가 날아가는** 경우가 많다.
> 학습 결과·체크포인트는 반드시 **영구 볼륨(persistent volume)** 이나 외부(HF Hub 등)에 저장한다.
> RunPod이면 `/workspace`가 영구 볼륨이니 거기서 작업한다.

---

## 2. 서버에 Claude Code 설치 + 구독 토큰 연결

SSH로 서버에 들어온 상태에서.

### 2-1. Node.js + Claude Code
```bash
# Node 20+ 없으면
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code
claude --version
```

### 2-2. 구독 토큰 — API 아님, 정액 구독으로 인증
서버는 브라우저가 없으니 **브라우저 되는 노트북**에서 토큰을 발급해 옮긴다.

```bash
# [노트북에서] 브라우저로 Claude 로그인 → 장기 토큰 출력
claude setup-token
```
```bash
# [서버에서] 그 토큰을 환경변수로 고정
echo 'export CLAUDE_CODE_OAUTH_TOKEN="발급받은-토큰"' >> ~/.bashrc
source ~/.bashrc
```

- 이 토큰은 **본인 구독(Pro/Max)에 묶인다.** 그 친구가 자기 계정·자기 브라우저로 발급해야 한다.
- **토큰당 과금 없음.** API 키(`ANTHROPIC_API_KEY`)를 쓰면 쓴 만큼 청구되니, 구독형이면 이걸 쓰지 말 것.

### 2-3. 확인
```bash
cd ~ && claude      # 프롬프트 뜨고 대화되면 성공. /exit 로 나감
```

---

## 3. tmux — 노트북 꺼져도 살아있는 세션 (제일 중요)

SSH가 끊기면 그 안의 작업은 죽는다. tmux로 서버에 세션을 상주시킨다.

```bash
sudo apt-get install -y tmux

tmux new -s train        # 'train' 세션 생성
#   ... 이 안에서 학습·Claude를 돌린다 ...
# 빠져나오기(세션은 살아있음): Ctrl+b 누르고 d

# 노트북 재접속 후 다시 붙기:
tmux attach -t train

# 세션 목록:
tmux ls
```

> **원리:** tmux 세션은 SSH가 아니라 **서버**에 붙어 있다. 그래서 노트북을 끄든 와이파이가
> 끊기든, 서버만 켜져 있으면 세션 안의 학습은 계속 돈다. 로컬 WSL이 재부팅되면 서비스가
> 다 내려갔던 것과 정반대 — 여기선 서버가 안 꺼지니 유지된다.

권장 배치: tmux 창을 둘로 나눠 **하나는 학습, 하나는 Claude**.
`Ctrl+b` → `"` (가로 분할) 또는 `%` (세로 분할).

---

## 4. Qwen 3B 수학 튜닝 환경 (Claude에게 시킬 것)

tmux 안에서 `claude`를 띄우고 아래를 순서대로 지시하면 된다. Claude가 실제 명령을 짠다.

1. **작업 폴더 + 파이썬 환경**
   ```bash
   mkdir -p ~/work && cd ~/work    # 대여 서버면 ~/work 대신 /workspace
   ```
   → "uv로 가상환경 만들고 Unsloth·TRL·datasets 설치해줘"

2. **튜닝 도구 — Unsloth 권장**
   - **Unsloth**: 단일 GPU 메모리 효율 최고. Qwen2.5/Qwen3 QLoRA에 딱. (대안: Axolotl, LLaMA-Factory, TRL)
   - 4비트 QLoRA면 3B가 8~10GB에 들어간다.

3. **베이스 모델 + 수학 데이터셋**
   - 모델: `Qwen/Qwen2.5-3B`(또는 대회가 지정한 3B) — `huggingface-cli download`
   - 데이터: GSM8K, MATH, MetaMathQA 등 수학 특화 셋
   - 함정: 일부 모델/데이터는 gated(승인 필요)라 HF 토큰이 필요하다. 미리 `huggingface-cli login`.

4. **학습 실행** — 아래 5장 방식으로 tmux 안에서 백그라운드로.

5. **평가** — 튜닝 후 GSM8K 정확도 등으로 검증하고, 좋으면 어댑터를 병합/업로드.

> 대회 규정(허용 베이스 모델, 데이터 출처, 제출 형식)을 먼저 확인하고 Claude에게 알려주면
> 그에 맞춰 파이프라인을 짠다.

---

## 5. "알아서 돌아가게" — 학습과 Claude의 역할 분리

핵심 이해: **학습 자체는 한 번 시작하면 Claude 없이도 몇 시간~며칠 혼자 돈다.**
Claude의 역할은 "환경 세팅 → 학습 실행 → 결과(loss·정확도) 읽고 → 설정 바꿔 재실행"의 반복이다.

그래서 실전 배치는 이렇게 한다:

### 학습 프로세스는 로그 파일로 흘리고 백그라운드로
```bash
# tmux 안에서. nohup + & 로 띄우면 이 창을 닫아도 학습은 계속
nohup python train_qwen_math.py > ~/work/train.log 2>&1 &
echo $! > ~/work/train.pid        # PID 저장

# 진행 상황 실시간으로 보기
tail -f ~/work/train.log
nvidia-smi                         # GPU 사용률 확인
```
이렇게 하면 **학습은 Claude나 SSH와 완전히 무관하게** 돈다. 노트북 꺼도 무방.

### Claude는 필요할 때 붙어서 다음 실험을 지시
- 학습이 끝나면(또는 중간 로그를 보고) Claude에게 "train.log 읽고 loss 추이 분석해서
  learning rate 낮춰 재실행해줘" 식으로 시킨다.
- 이러면 **구독 사용량 한도**를 아낄 수 있다. Claude를 몇 시간 쉬지 않고 자율로 돌리면
  구독 한도(주기적 리셋)에 걸려 잠깐 멈출 수 있는데, 학습은 어차피 스스로 도니
  Claude는 "실험 사이사이"에만 쓰면 된다.

### 완전 무인으로 여러 실험을 돌리고 싶다면
사람 확인 없이 한 번에 시키는 헤드리스 모드:
```bash
claude -p "train.log를 분석해 다음 하이퍼파라미터를 정하고 학습을 재실행한 뒤 결과를 요약해줘" \
  --dangerously-skip-permissions
```
> ⚠️ `--dangerously-skip-permissions`는 파일 삭제·명령 실행을 **묻지 않고** 한다.
> **날아가도 되는 실험용 서버에서만** 쓴다. 중요 데이터가 있는 서버에선 쓰지 말 것.
> 그리고 무인 자율 실행은 구독 한도에 더 빨리 닿는다. 처음엔 붙어서 지켜보며 돌리길 권한다.

---

## 6. 결과 안전하게 지키기

- 대여 서버(RunPod·Vast)는 인스턴스가 사라질 수 있다. 체크포인트를 **영구 볼륨**(`/workspace`)에 두거나
  중간중간 **HF Hub에 push**(`huggingface-cli upload`)해 백업한다.
- 학습이 오래 걸리면 Claude에게 "N 스텝마다 체크포인트 저장하도록 설정해줘"라고 시킨다.
  중간에 서버가 죽어도 이어서 재개할 수 있다.

---

## 체크리스트

- [ ] 서버가 **항상 켜져 있는지** 확인 (본인 노트북 ❌)
- [ ] `nvidia-smi`로 GPU·VRAM 확인 (16GB↑ 권장)
- [ ] Node + Claude Code 설치
- [ ] 노트북에서 `claude setup-token` → 서버에 `CLAUDE_CODE_OAUTH_TOKEN`
- [ ] `tmux new -s train`으로 상주 세션
- [ ] Unsloth 등으로 Qwen 3B QLoRA 환경 구성 (Claude에게)
- [ ] 학습은 `nohup ... &` + 로그 파일로 백그라운드
- [ ] 체크포인트를 영구 볼륨/HF Hub에 백업
- [ ] 노트북 꺼보고 → 재접속 → `tmux attach`로 계속됨을 확인

## 안 헷갈리게 요약

| 오해 | 사실 |
|---|---|
| "AWS 세션이 있어야 계속 돈다" | 아니다. **안 꺼지는 서버 + tmux**면 된다. AWS는 그런 서버를 빌리는 한 방법일 뿐 |
| "노트북이 계속 켜져 있어야 한다" | 아니다. 노트북은 화면일 뿐. 학습·Claude는 **서버에서** 돈다 |
| "Claude를 24시간 돌려야 학습이 된다" | 아니다. 학습은 스스로 돈다. Claude는 실험을 설계·수정하는 사이사이에만 |
| "구독으로는 서버에서 못 쓴다" | 된다. `claude setup-token`으로 헤드리스 인증 |
