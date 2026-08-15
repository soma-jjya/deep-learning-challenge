#!/bin/bash
# exp58 — 시스템 프롬프트 단일 변형 스윕 (H27). 생성 후 자동 채점까지.
#
# 대조군(base)을 **같은 실행 안에서 새로 생성**하는 것이 이 실험의 핵심 설계다.
# 기존 75.8%와 비교하면 vLLM 비결정성이 섞여 ±1%p가 그냥 붙는다(exp34b가 실측한 재실행 변동).
# 같은 실행·같은 시드 안에서 프롬프트만 다르면 그 차이는 온전히 프롬프트의 것이다.
#
# 사용: setsid nohup bash remote/run_exp58_chain.sh > exp58_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

for i in $(seq 1 240); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "${FREE:-0}" -gt 18000 ] && [ "$APPS" -eq 0 ]; then break; fi
  sleep 30
done
echo "[$(date +%H:%M)] GPU 여유 $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

uv run python remote/sweep_system_prompt.py --n 32 --seed 42

echo
echo "===== exp58 채점 (같은 실행·같은 시드, 프롬프트만 다름) ====="
for NM in base verify classify direct extract; do
  F=results/val_samples_sp_${NM}_s42.jsonl
  [ -s "$F" ] || { echo "  [$NM] 덤프 없음"; continue; }
  printf '%-10s ' "$NM"
  uv run python remote/analyze_selection_gap.py --samples "$F" --n 32 | sed -n '2p'
done
echo "[$(date +%H:%M)] EXP58_DONE"
echo "판정: 같은 실행의 base 대비 +5문항 이상이면 시드 43·44로 확장, 이어서 리더보드 확인."
