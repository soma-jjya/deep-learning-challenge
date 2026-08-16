#!/bin/bash
# exp68 — H31 경계 난이도 구간 SFT. 학습 8전째, 그러나 데이터 선택 원리가 처음으로 다르다.
#
# 학습 7전의 데이터는 두 극단이었다:
#   · SFT 3전 : "베이스가 푼 문제 전부" → 실측 분포상 54.7%가 이미 4/4로 맞히는 문제였다.
#               모델이 아는 것을 다시 먹이면 기존 행동 쪽으로 날카로워지고,
#               어려운 문제에서는 기존 **오답** 쪽으로 날카로워진다.
#   · 교사증류 : "전혀 못 푸는 0/32 문제" → 능력 부재 구간. pass@32는 올랐으나 투표는 졌다.
# 가운데(1~3/4로 가끔 맞히는 1,464문항)는 SFT로 한 번도 안 겨냥했다.
# 그 구간이 병목의 정확한 좌표다 — 정답이 이미 표본에 나타나지만 표결에서 밀린다.
#
# 유보: 같은 문항 집합을 exp47 DPO가 썼다(선호쌍 1,464개). 다만 DPO는 상대 순위 학습이고
# lr 5e-6·beta 0.3의 매우 보수적 설정이었으며 rewards/accuracies 0.60으로 신호가 약했다.
# 여기서는 정답 경로를 직접 모방시킨다.
#
# 에폭 2인 이유: 2,476샘플 / 유효배치 16 = 에폭당 155스텝. 2에폭 310스텝이면 각 샘플을
# 두 번 본다(표준). exp53b의 교훈(작은 데이터에 스텝을 늘리면 외운다)을 감안해 3은 피한다.
# 손실 마스킹은 exp59에서 고친 상태(assistant 구간만)를 그대로 쓴다.
#
# 예산: 학습 ~50분 + 평가 2시드 × 45분 = 약 2.3시간. 두 시드 모두 양수면 시드 44 추가.
#
# 사용: setsid nohup bash remote/run_exp68_chain.sh > exp68_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

OUTDIR=outputs/qlora_marginal
ADAPTER=$OUTDIR/qlora_r16_lr5e-05_ep2_final

wait_gpu() {
  for i in $(seq 1 720); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "${FREE:-0}" -gt 18000 ] && [ "$APPS" -eq 0 ]; then return 0; fi
    sleep 30
  done
}

echo "[$(date +%H:%M)] 앞선 작업(exp64) 종료 대기"
while pgrep -f 'sweep_temp_and_promp[t].py|dump_lb_sample[s].py' > /dev/null; do sleep 60; done
wait_gpu

echo "[$(date +%H:%M)] 학습 (경계구간 2,476풀이, 2에폭, assistant 구간만 손실)"
uv run python remote/train_qlora.py --data-path data/sft_marginal_13.jsonl \
  --output-dir "$OUTDIR" --epochs 2 > train68.log 2>&1

grep -E '손실 마스킹 확인|⛔' train68.log || true
if [ ! -d "$ADAPTER" ]; then
  echo "⛔ 어댑터 없음: $ADAPTER"; tail -30 train68.log; exit 1
fi

for S in 42 43; do
  OUT=results/val_samples_marginal_s$S.jsonl
  [ -s "$OUT" ] && { echo "[$(date +%H:%M)] seed $S 이미 있음"; continue; }
  wait_gpu
  echo "[$(date +%H:%M)] seed $S 덤프"
  uv run python remote/dump_samples.py --n 32 --seed "$S" --adapter "$ADAPTER" --out "$OUT"
done

echo
echo "===== exp68 판정: 베이스 vs 경계구간 SFT ====="
for S in 42 43; do
  case $S in 42) BASE=results/val_samples.jsonl ;; *) BASE=results/val_samples_s$S.jsonl ;; esac
  printf 'seed %s 베이스   ' "$S"
  uv run python remote/analyze_selection_gap.py --samples "$BASE" --n 32 | sed -n '2p'
  printf 'seed %s 경계SFT  ' "$S"
  uv run python remote/analyze_selection_gap.py --samples "results/val_samples_marginal_s$S.jsonl" --n 32 | sed -n '2p'
  printf 'seed %s pass@32  베이스/경계  ' "$S"
  uv run python remote/analyze_selection_gap.py --samples "$BASE" --n 32 | sed -n '7p' | tr -d '\n'
  uv run python remote/analyze_selection_gap.py --samples "results/val_samples_marginal_s$S.jsonl" --n 32 | sed -n '7p'
done
echo "[$(date +%H:%M)] EXP68_DONE"
echo "판정: 두 시드 모두 양수이고 합이 +8문항 이상이면 시드 44 추가. 아니면 종결."
echo "⚠️ 483문항 임계는 ±1.49%p(약 7문항)다. 그보다 작으면 '효과 없음'이 아니라 '측정 불가'로 기록할 것."
