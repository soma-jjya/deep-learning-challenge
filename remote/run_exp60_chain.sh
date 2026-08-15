#!/bin/bash
# exp60 — 마스킹을 고치고 **exp06c를 그대로** 재실행한다. 학습 축 재개의 본 실험.
#
# 왜 exp06c인가: 초기 SFT 3전이 "학습은 해롭다"는 이 프로젝트 결론의 출발점이었고,
# 그 실험들이 쓴 RFT 데이터는 **잘못 걸린 손실 비중이 19.4%** 다(교사 데이터는 8.4%).
# 교사 데이터에서 8.4%가 6문항을 먹었으므로(exp59 시드42: −9 → −3), 19.4%면 더 클 수 있다.
# exp06c 기준값: greedy 67.9% / SC 73.3% (베이스 69.4% / 74.7%).
#
# 조건은 exp06c와 동일하게 둔다 — data/sft_short.jsonl, lr 5e-5, 1에폭, r16.
# 마스킹만 다르다. 그래야 차이가 온전히 마스킹의 것이다.
#
# 판정: exp48 원칙 — 3시드 부호 일관 + 평균 +1.5%p 이상. 최종 스택 변경은 사용자 승인 사항.
#
# 사용: setsid nohup bash remote/run_exp60_chain.sh > exp60_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

OUTDIR=outputs/qlora_rft_masked
ADAPTER=$OUTDIR/qlora_r16_lr5e-05_ep1_final

wait_gpu() {
  for i in $(seq 1 480); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "${FREE:-0}" -gt 18000 ] && [ "$APPS" -eq 0 ]; then return 0; fi
    sleep 30
  done
}

echo "[$(date +%H:%M)] 앞선 작업(exp59) 종료 대기"
while pgrep -f 'dump_sample[s].py|train_qlor[a].py' > /dev/null; do sleep 60; done
wait_gpu

echo "[$(date +%H:%M)] 학습 시작 (RFT 최단 풀이, 1에폭, assistant 구간만 손실)"
uv run python remote/train_qlora.py --data-path data/sft_short.jsonl \
  --output-dir "$OUTDIR" --epochs 1 > train60.log 2>&1

grep -E '손실 마스킹 확인|⛔' train60.log || true
if [ ! -d "$ADAPTER" ]; then
  echo "⛔ 어댑터 없음: $ADAPTER — 학습 실패. 중단"; tail -30 train60.log; exit 1
fi

for S in 42 43 44; do
  OUT=results/val_samples_rftmask_s$S.jsonl
  [ -s "$OUT" ] && { echo "[$(date +%H:%M)] seed $S 이미 있음"; continue; }
  wait_gpu
  echo "[$(date +%H:%M)] seed $S 덤프"
  uv run python remote/dump_samples.py --n 32 --seed "$S" --adapter "$ADAPTER" --out "$OUT"
done

echo
echo "===== exp60 판정: 베이스 vs RFT 학습(마스킹 후) ====="
for S in 42 43 44; do
  case $S in 42) BASE=results/val_samples.jsonl ;; *) BASE=results/val_samples_s$S.jsonl ;; esac
  printf 'seed %s 베이스     ' "$S"
  uv run python remote/analyze_selection_gap.py --samples "$BASE" --n 32 | sed -n '2p'
  printf 'seed %s RFT마스킹  ' "$S"
  uv run python remote/analyze_selection_gap.py --samples "results/val_samples_rftmask_s$S.jsonl" --n 32 | sed -n '2p'
done
echo "[$(date +%H:%M)] EXP60_DONE"
echo "참고 기준: exp06c(마스킹 전, 같은 데이터·같은 lr·1에폭) SC n8 73.3% / 베이스 74.7%"
echo "판정: 3시드 부호 일관 + 평균 +1.5%p 이상일 때만 채택. 최종 스택 변경은 사용자 승인 사항."
