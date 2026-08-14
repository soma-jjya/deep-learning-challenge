#!/bin/bash
# 유휴 자동 종료 감시자 — cron이 5분마다 실행 (setup_env.sh가 등록)
# GPU가 놀고 실험 프로세스도 없는 상태가 30분 이어지면 인스턴스를 stop한다.
# 예외: ~/KEEP_ALIVE 파일이 있으면 절대 끄지 않는다 (수동 작업 시: touch ~/KEEP_ALIVE)
STATE=/tmp/idle_count
source /home/ubuntu/.ajudl_env 2>/dev/null || true   # NTFY_TOPIC 로드 (cron에는 env가 없음)

if [ -f /home/ubuntu/KEEP_ALIVE ]; then echo 0 > $STATE; exit 0; fi

busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -nr | head -1)
busy=${busy:-0}
# 실제 작업 프로세스만 센다 — claude/runner가 유령 상태로 매달려도 유휴면 정지되게
# (정지돼도 부팅 자동시작이 러너를 되살리므로 안전)
#
# ⚠️ 2026-08-11 수정: 예전에는 스크립트 이름을 하드코딩했다
#   ("train_qlora|generate_rft|eval_vllm|make_submission|prep_numina").
#   그래서 새로 만든 train_reward_head.py가 목록에 없어 **학습 중인데도 유휴로 세어
#   인스턴스를 꺼뜨렸다**(exp51 holdout 채점 중 사고). 새 스크립트를 만들 때마다
#   목록을 고쳐야 하는 구조 자체가 잘못이므로, remote/ 아래 파이썬 작업 전반과
#   체인 스크립트를 포괄하는 패턴으로 바꾼다.
#
# ⚠️ 2026-08-14 재수정: 같은 함정에 또 빠질 뻔했다. 위 패턴은 `remote/`만 보는데
#   교사 풀이 생성은 `api/gen_teacher_claude_code.py`라 **4시간째 돌고 있는데도 유휴로
#   세어졌다**. 인스턴스가 살아남은 건 큐에 미완료 항목이 있어 러너 부활 분기가 카운터를
#   초기화해준 우연 덕분이었고, 큐가 비는 순간 생성 도중에 꺼졌을 것이다.
#   디렉토리를 열거하는 대신 **이 저장소의 작업 스크립트 전반**을 잡도록 넓힌다.
procs=$(pgrep -cf "python +(remote|api)/|(remote|api)/[a-z_]+\.py|run_.*_chain\.sh|run_paired_seeds\.sh|teacher_supervisor\.sh" || true)

if [ "$busy" -gt 10 ] || [ "$procs" -gt 0 ]; then
  echo 0 > $STATE
  exit 0
fi

# ── 러너 자동 부활 (2026-08-10 추가) ──
# 작업 프로세스가 없는 지금, 큐에 할 일이 남았는데 러너까지 죽었다면 되살린다.
# (2026-08-09에 러너가 2회 죽어 3시간을 유휴로 낭비한 사고의 근본 대책)
REPO=/home/ubuntu/work/deep-learning-challenge
if [ -d "$REPO" ] && grep -q '^- \[ \]' "$REPO/experiments/queue.md" 2>/dev/null; then
  if ! pgrep -f "run_experiment[s].sh" > /dev/null; then
    logger "ajudl watchdog: 큐에 할 일이 있는데 러너 없음 — 재시작"
    [ -n "$NTFY_TOPIC" ] && curl -s -d "러너 사망 감지 — 자동 부활" "ntfy.sh/$NTFY_TOPIC" > /dev/null
    cd "$REPO" || exit 0
    echo "=== watchdog revival $(date -u) ===" >> runner.log
    sudo -u ubuntu env PATH="/home/ubuntu/.local/bin:$PATH" \
      nohup setsid bash remote/run_experiments.sh >> runner.log 2>&1 < /dev/null &
    echo 0 > $STATE   # 부활시켰으니 유휴 카운터 초기화
    exit 0
  fi
fi

n=$(cat $STATE 2>/dev/null || echo 0)
n=$((n + 1))
echo $n > $STATE

if [ "$n" -ge 6 ]; then   # 5분 x 6 = 30분 유휴
  logger "ajudl watchdog: 30분 유휴 — 인스턴스 정지"
  [ -n "$NTFY_TOPIC" ] && curl -s -d "GPU 서버 유휴 30분 → 자동 정지" "ntfy.sh/$NTFY_TOPIC" > /dev/null
  sudo shutdown -h now   # launch.ps1이 shutdown-behavior=stop으로 설정 → stop됨 (디스크 유지)
fi
