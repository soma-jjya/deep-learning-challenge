#!/bin/bash
# exp53 전체 체인 — 학습이 끝나기를 기다렸다가 곧바로 평가 체인으로 넘어간다.
#
# 왜 서버에서 이어붙이나: 로컬에서 완료를 기다리는 방식은 세션이 끊기면 같이 끊긴다.
# 학습(약 18분)과 덤프 3회(약 1.8시간) 사이에 GPU를 놀리지 않으려면 서버가 스스로
# 이어가야 한다.
#
# ⚠️ pgrep 패턴에 대괄호를 넣는 이유: `pgrep -f`는 명령줄 전체를 보므로, 패턴을 그대로
# 쓰면 이 스크립트를 띄운 명령줄이나 감시 루프 자신이 매칭돼 영원히 안 풀린다
# (2026-08-13 PRM 체인이 정확히 이 함정으로 교착됐다). `train_qlor[a]`는 실제
# 프로세스명 train_qlora.py에는 맞고, 이 파일을 읽는 명령줄에는 맞지 않는다.
#
# 사용: setsid nohup bash remote/run_exp53_chain.sh > exp53_chain.log 2>&1 & disown
set -u
cd $HOME/work/deep-learning-challenge
ADAPTER=outputs/qlora_teacher/qlora_r16_lr5e-05_ep3_final

echo "[$(date +%H:%M)] 학습 종료 대기"
while pgrep -f 'train_qlor[a].py' > /dev/null; do sleep 60; done
echo "[$(date +%H:%M)] 학습 프로세스 종료 확인"

if [ ! -d "$ADAPTER" ]; then
  echo "⛔ 어댑터가 없다: $ADAPTER — 학습이 실패했을 가능성. 체인 중단"
  tail -20 train53_teacher.log
  exit 1
fi

echo "[$(date +%H:%M)] 평가 체인 시작"
bash remote/run_teacher_eval_chain.sh "$ADAPTER"
echo "[$(date +%H:%M)] 체인 종료"
