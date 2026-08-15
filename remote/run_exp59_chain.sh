#!/bin/bash
# exp59 — 손실 마스킹을 고치고 exp53c를 그대로 재실행한다.
#
# 왜 exp53c와 같은 조건인가: 데이터(교사 683개)·에폭(1)·lr(5e-5)·r(16)·시드(42/43/44)를
# 전부 동일하게 두고 **마스킹만** 바꾼다. 그래야 차이가 온전히 마스킹의 것이다.
# exp53c 기준값: 시드별 −9 / −7 / −10문항, 평균 −1.79%p.
#
# 무엇이 틀려 있었나: messages를 텍스트로 렌더링해 dataset_text_field로 넘기면 TRL은
# 그것을 language modeling 데이터셋으로 보고 **전체 시퀀스**에 손실을 건다. 시스템
# 프롬프트와 문제 본문까지 학습 대상이 됐다는 뜻이다. SFT 4전이 전부 이 설정이었다.
#
# 판정: exp48 원칙 — 3시드 부호 일관 + 평균 +1.5%p 이상. 최종 스택 변경은 사용자 승인 사항.
#
# ⚠️ GPU 반환은 여유 메모리로 판정한다(vLLM EngineCore 자식이 부모보다 오래 산다).
# ⚠️ pgrep 패턴의 대괄호는 자기 자신 매칭 방지.
#
# 사용: setsid nohup bash remote/run_exp59_chain.sh > exp59_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

OUTDIR=outputs/qlora_teacher_masked
ADAPTER=$OUTDIR/qlora_r16_lr5e-05_ep1_final

wait_gpu() {
  for i in $(seq 1 240); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "${FREE:-0}" -gt 18000 ] && [ "$APPS" -eq 0 ]; then return 0; fi
    if [ "$i" -gt 3 ] && ! pgrep -f 'dump_sample[s].py|sweep_system_promp[t].py' > /dev/null; then
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
        echo "  고아 GPU 프로세스 정리: $p"; kill "$p" 2>/dev/null || true
      done
    fi
    sleep 30
  done
}

wait_gpu
echo "[$(date +%H:%M)] 학습 시작 (교사 683개, 1에폭, assistant 구간만 손실)"
uv run python remote/train_qlora.py --data-path data/sft_teacher_full.jsonl \
  --output-dir "$OUTDIR" --epochs 1 > train59.log 2>&1

if [ ! -d "$ADAPTER" ]; then
  echo "⛔ 어댑터 없음: $ADAPTER — 학습 실패. 중단"; tail -30 train59.log; exit 1
fi
grep -E '손실은 assistant|전체 시퀀스' train59.log || true

for S in 42 43 44; do
  OUT=results/val_samples_masked_s$S.jsonl
  [ -s "$OUT" ] && { echo "[$(date +%H:%M)] seed $S 이미 있음"; continue; }
  wait_gpu
  echo "[$(date +%H:%M)] seed $S 덤프"
  uv run python remote/dump_samples.py --n 32 --seed "$S" --adapter "$ADAPTER" --out "$OUT"
done

echo
echo "===== exp59 판정: 베이스 / exp53c(마스킹 전) / exp59(마스킹 후) ====="
for S in 42 43 44; do
  case $S in 42) BASE=results/val_samples.jsonl ;; *) BASE=results/val_samples_s$S.jsonl ;; esac
  echo "--- seed $S 베이스"
  uv run python remote/analyze_selection_gap.py --samples "$BASE" --n 32 | sed -n '2p'
  echo "--- seed $S exp53c (전체 시퀀스 손실)"
  uv run python remote/analyze_selection_gap.py --samples "results/val_samples_teacher_full_s$S.jsonl" --n 32 | sed -n '2p'
  echo "--- seed $S exp59 (assistant 구간만)"
  uv run python remote/analyze_selection_gap.py --samples "results/val_samples_masked_s$S.jsonl" --n 32 | sed -n '2p'
done
echo "[$(date +%H:%M)] EXP59_DONE"
echo "판정: 3시드 부호 일관 + 평균 +1.5%p 이상일 때만 채택. 최종 스택 변경은 사용자 승인 사항."
