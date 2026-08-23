#!/bin/bash
# exp93 — NuminaMath 대규모 full FT (구 exp87 재개). 사전등록: prereg_exp93_bigft.md
#
# 체인: 데이터 필터링 → full FT (ckpt 4개) → ckpt별 val 덤프 (n=32, seed 42)
# 채점은 로컬에서 교정 정답지로 한다 (서버는 덤프만).
# 하드웨어 전제: g6e.xlarge (L40S 48GB). OOM 시 train_fullft.py 머리말 참조.
#
# 사용: setsid nohup bash remote/run_exp93_chain.sh > exp93_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

notify() { curl -s -d "$1" "https://ntfy.sh/${NTFY_TOPIC:-none}" >/dev/null 2>&1 || true; }
stamp() { echo "[$(date +%m/%d\ %H:%M)] $1" | tee -a exp93_status.txt; }

stamp "exp93 체인 시작 (GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"

# 1) 데이터 필터링 (이미 있으면 건너뜀)
if [ ! -s data/numina_full.jsonl ]; then
  stamp "1/3 데이터 필터링 시작"
  uv run python remote/prep_numina_full.py > exp93_prep.log 2>&1 || {
    stamp "❌ 필터링 실패"; notify "exp93 필터링 실패"; exit 1; }
fi
stamp "1/3 데이터 준비 완료: $(wc -l < data/numina_full.jsonl)개"

# 2) full FT
if [ ! -d outputs/exp93_fullft/final ]; then
  stamp "2/3 full FT 시작"
  uv run python remote/train_fullft.py > exp93_train.log 2>&1 || {
    stamp "❌ 학습 실패 (exp93_train.log 확인)"; notify "exp93 학습 실패"; exit 1; }
fi
stamp "2/3 학습 완료"
notify "exp93 학습 완료, 덤프 시작"

# 3) 체크포인트별 val 덤프
for CKPT in $(ls -d outputs/exp93_fullft/checkpoint-* | sort -t- -k2 -n); do
  STEP=$(basename "$CKPT" | cut -d- -f2)
  OUT=results/val_samples_e93_c${STEP}_s42.jsonl
  if [ -s "$OUT" ]; then continue; fi
  stamp "3/3 덤프: ckpt-$STEP"
  uv run python remote/dump_samples.py --model "$CKPT" --n 32 --seed 42 \
    --out "$OUT" > exp93_dump_${STEP}.log 2>&1 || {
    stamp "❌ 덤프 실패: ckpt-$STEP"; notify "exp93 덤프 실패 ckpt-$STEP"; exit 1; }
done

stamp "✅ exp93 체인 완료 — 로컬에서 채점하세요"
notify "exp93 체인 완료 (덤프까지)"
