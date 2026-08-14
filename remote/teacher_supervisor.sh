#!/bin/bash
# 교사 풀이 생성을 끝까지 밀어붙이는 감시 루프.
#
# 왜 필요한가: `claude -p`는 사용 한도에 걸리면 stdout·stderr가 빈 채로 즉시 돌아온다.
# 워커는 연속 실패를 감지하면 스스로 멈추도록 고쳤으므로(gen_teacher_claude_code.py의
# --give-up-after), 한도가 풀린 뒤 누군가 다시 띄워줘야 한다. 이 스크립트가 그 역할이다.
# 재개는 이미 만든 출력물의 '## ID:' 헤더로 판정하므로 몇 번을 다시 띄워도 중복 생성이 없다.
#
# 병렬 수: 16으로 올렸더니 31분 만에 한도에 걸렸다(2026-08-14 실측). 6은 그보다 오래 버티고,
# 어차피 한도가 총량이라면 천천히 가는 쪽이 창을 헛되이 태우지 않는다.
#
# 사용: setsid nohup bash remote/teacher_supervisor.sh > teacher_sup.log 2>&1 & disown
set -u
WORKERS=${1:-6}
TARGET=${2:-1200}
WAIT=${3:-1800}          # 한 바퀴 끝나고 다음 시도까지 (한도 회복 대기)

cd $HOME/work/deep-learning-challenge

while true; do
  DONE=$(grep -ho '^## ID: .*' data/teacher_out/*.txt 2>/dev/null | sort -u | wc -l)
  echo "[$(date +%H:%M)] 완료 $DONE/$TARGET"
  if [ "$DONE" -ge "$TARGET" ]; then
    echo "목표 도달 — 감시 종료"
    break
  fi

  if ! pgrep -f 'gen_teacher_claude[_]code' > /dev/null; then
    echo "[$(date +%H:%M)] 워커 $WORKERS개 기동"
    bash remote/launch_teacher_workers.sh "$WORKERS"
  fi

  sleep "$WAIT"
done
