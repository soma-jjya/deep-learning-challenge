#!/bin/bash
# 자율 실험 러너 — 서버의 tmux 안에서 실행:
#   nohup bash remote/run_experiments.sh > runner.log 2>&1 &
# experiments/queue.md의 미완료([ ]) 실험을 위에서부터 하나씩 Claude에게 실행시킨다.
# 큐가 비면 종료 → watchdog이 30분 뒤 인스턴스를 자동 stop.
set -e
cd ~/work/deep-learning-challenge

notify() {
  [ -n "$NTFY_TOPIC" ] && curl -s -d "$1" "ntfy.sh/$NTFY_TOPIC" > /dev/null || true
}

notify "실험 러너 시작"

while true; do
  git pull --rebase
  if ! grep -q '^- \[ \]' experiments/queue.md; then
    echo "큐가 비었습니다. 러너 종료."
    notify "실험 큐 완료 — 서버는 곧 자동 정지됩니다"
    break
  fi

  NEXT=$(grep -m1 '^- \[ \]' experiments/queue.md)
  echo "=== 실행: $NEXT ==="
  notify "실험 시작: $NEXT"

  claude -p "당신은 GPU 서버의 실험 실행자입니다. CONTEXT.md와 experiments/queue.md를 읽으세요.
큐의 첫 번째 미완료 실험(- [ ])을 정확히 그 명세대로 실행하세요. 학습·생성은 이미 있는 remote/ 스크립트를 사용하고, 완료까지 기다리세요.
끝나면: ① 결과 수치를 EXPERIMENTS.md 표와 해당 실험 섹션에 기록 ② queue.md에서 해당 항목을 [x]로 바꾸고 결과 한 줄 덧붙임 ③ git add -A && git commit && git push
규칙: queue에 없는 실험을 임의로 추가하지 말 것. 베이스 모델은 Qwen2.5-3B-Instruct 고정. 검증 세트(val, seed=123)는 학습에 절대 사용 금지." \
    --dangerously-skip-permissions --max-turns 200 || {
      notify "실험 실패: $NEXT — runner.log 확인 필요"
      echo "Claude 실행 실패. 10분 후 재시도."
      sleep 600
      continue
    }

  git pull --rebase
  notify "실험 완료: $NEXT"
done
