#!/bin/bash
# exp64 — DPO 어댑터를 리더보드로 판정한다. 그리고 이어서 온도·프롬프트 스윕.
#
# 왜 DPO를 이제서야 LB에 올리나: exp47에서 같은 어댑터를 두 번 재서 75.2% / 76.6%로
# 1.4%p 어긋났고, exp48의 시드 대응 비교가 +4/+1/−7(평균 −0.14%p)로 부호 혼재 =
# 노이즈로 확정했다. 그래서 **로컬에서 멈췄다.** 그런데 우리 정책은
# "로컬은 오답의 23%가 라벨 의심인 흐린 자, 리더보드 831문항은 운영진 검수를 거친 깨끗한 자"
# 이므로 구조적으로 다른 후보는 LB로 확인하게 되어 있다. exp41은 로컬이 좋아 보였는데
# LB가 정반대였고, 그때 LB를 심판으로 삼았다. 이번엔 그 심판에게 묻지 않은 것이 누락이다.
# 게다가 exp52b로 우리 검증셋에 실제 라벨 오류가 있음을 확인했으므로, 깨끗한 자에서
# 다른 답이 나올 여지가 있다.
#
# 판정: LB 점수만 본다. 기존 최고 0.78459 대비 재실행 변동(0.48%p)을 넘어야 의미가 있다.
# 최종 스택 변경은 사용자 승인 사항.
#
# 사용: setsid nohup bash remote/run_exp64_chain.sh > exp64_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

ADAPTER=outputs/dpo/dpo_r8_b0.3_lr5e-06_final
DUMP=results/lb_samples_dpo.jsonl

wait_gpu() {
  for i in $(seq 1 480); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "${FREE:-0}" -gt 18000 ] && [ "$APPS" -eq 0 ]; then return 0; fi
    sleep 30
  done
}

wait_gpu
echo "[$(date +%H:%M)] DPO 어댑터로 리더보드 831문항 × 32샘플 덤프"
uv run python remote/dump_lb_samples.py --n 32 --seed 42 --adapter "$ADAPTER" --out "$DUMP"

echo "[$(date +%H:%M)] 제출 파일 생성"
uv run python remote/make_submission_from_dump.py --dump "$DUMP" --rule weighted --n 32 --tag dpolb

echo "[$(date +%H:%M)] EXP64_DUMP_DONE — 로컬에서 검증 후 제출할 것"

# GPU가 남았으니 이어서 온도·프롬프트 스윕(H29·H30)
wait_gpu
echo "[$(date +%H:%M)] 온도·프롬프트 스윕 시작 (대조군 포함 5종, 한 세션)"
uv run python remote/sweep_temp_and_prompt.py --n 32 --seed 42

echo
echo "===== 채점 (같은 실행·같은 시드) ====="
for NM in base_t07 base_t03 base_t11 minimal_t07 nosys_t07; do
  F=results/val_samples_tp_${NM}_s42.jsonl
  [ -s "$F" ] || { echo "  [$NM] 덤프 없음"; continue; }
  printf '%-14s ' "$NM"
  uv run python remote/analyze_selection_gap.py --samples "$F" --n 32 | sed -n '2p'
done
echo "[$(date +%H:%M)] EXP64_DONE"
echo "판정: 대조군 base_t07 대비 +5문항 이상이면 시드 43·44로 확장."
