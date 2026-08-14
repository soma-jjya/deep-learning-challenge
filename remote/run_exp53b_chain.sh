#!/bin/bash
# exp53b — "가설이 틀렸나, 내 설정이 틀렸나"를 가르는 체인.
#
# exp53(3에폭)은 시드 42에서 −5.80%p로 크게 하락했다. 그런데 **pass@32는 424→423으로
# 그대로**였다. 정답을 만들어내는 능력은 멀쩡한데 고르지를 못하게 됐다는 뜻이고,
# 이는 과적합의 전형적인 모습이다. 큐 명세는 1에폭이었는데 데이터가 560개라
# "35스텝은 개입이 너무 약하다"고 판단해 3에폭으로 올린 것이 원인일 수 있다.
# 그래서 1에폭 버전을 같은 절차로 재서 두 원인을 분리한다.
#
# 시드 44 덤프를 건너뛰고 이 실험에 GPU를 쓴다: −5.80%p는 시드 노이즈(±1%p)의 5배라
# 부호 확인에는 시드 43 하나면 충분하고, 같은 시간이면 이쪽이 정보량이 훨씬 크다.
#
# ⚠️ pgrep 패턴의 대괄호는 자기 자신 매칭을 막기 위한 것이다(8/13 PRM 체인 교착).
#
# 사용: setsid nohup bash remote/run_exp53b_chain.sh > exp53b_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

ADAPTER=outputs/qlora_teacher_ep1/qlora_r16_lr5e-05_ep1_final

echo "[$(date +%H:%M)] 앞선 덤프(seed 43) 종료 대기 — GPU 경합 방지"
while pgrep -f 'dump_sample[s].py' > /dev/null; do sleep 60; done

# 시드 44 덤프가 이어서 뜨지 않도록 평가 체인을 먼저 정리한다
pkill -f 'run_teacher_eval_chai[n].sh' 2>/dev/null || true
sleep 5
while pgrep -f 'dump_sample[s].py' > /dev/null; do sleep 30; done

echo "[$(date +%H:%M)] 1에폭 학습 시작"
uv run python remote/train_qlora.py --epochs 1 --output-dir outputs/qlora_teacher_ep1 \
  > train53b_teacher_ep1.log 2>&1

if [ ! -d "$ADAPTER" ]; then
  echo "⛔ 어댑터 없음: $ADAPTER — 학습 실패. 중단"
  tail -20 train53b_teacher_ep1.log
  exit 1
fi

echo "[$(date +%H:%M)] seed 42 덤프 시작"
uv run python remote/dump_samples.py --n 32 --seed 42 --adapter "$ADAPTER" \
  --out results/val_samples_teacher_ep1_s42.jsonl

echo
echo "===== seed 42 비교 ====="
echo "--- 베이스"
uv run python remote/analyze_selection_gap.py --samples results/val_samples.jsonl --n 32 | head -7
echo "--- 교사 3에폭"
uv run python remote/analyze_selection_gap.py --samples results/val_samples_teacher_s42.jsonl --n 32 | head -7
echo "--- 교사 1에폭"
uv run python remote/analyze_selection_gap.py --samples results/val_samples_teacher_ep1_s42.jsonl --n 32 | head -7
echo "[$(date +%H:%M)] 체인 종료"
