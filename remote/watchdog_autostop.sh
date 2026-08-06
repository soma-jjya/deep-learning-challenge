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
procs=$(pgrep -cf "train_qlora|generate_rft|eval_vllm|make_submission|prep_numina" || true)

if [ "$busy" -gt 10 ] || [ "$procs" -gt 0 ]; then
  echo 0 > $STATE
  exit 0
fi

n=$(cat $STATE 2>/dev/null || echo 0)
n=$((n + 1))
echo $n > $STATE

if [ "$n" -ge 6 ]; then   # 5분 x 6 = 30분 유휴
  logger "ajudl watchdog: 30분 유휴 — 인스턴스 정지"
  [ -n "$NTFY_TOPIC" ] && curl -s -d "GPU 서버 유휴 30분 → 자동 정지" "ntfy.sh/$NTFY_TOPIC" > /dev/null
  sudo shutdown -h now   # launch.ps1이 shutdown-behavior=stop으로 설정 → stop됨 (디스크 유지)
fi
