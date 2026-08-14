#!/bin/bash
# exp53c — H23(강한 교사 증류) 최종 판정. 전량 데이터 × 1에폭 × 시드 3개.
#
# 왜 이 형태인가:
#   - 1에폭: exp53b가 3에폭 대비 손상을 −28문항에서 −9문항으로 줄였다. 큐 명세가 옳았고
#     "560개에 35스텝은 개입이 약하다"는 판단이 틀렸다. 가설은 가장 유리한 조건에서 재야 한다.
#   - 전량 데이터: 교사 생성이 1,200문제를 마치면 학습 데이터가 560 → 약 700개로 늘어난다.
#     이미 만들어둔 데이터라 추가 비용이 없다.
#   - 시드 3개: exp53b의 −1.86%p는 노이즈 폭의 2배 남짓이라 단일 시드로는 판정할 수 없다.
#     판정은 exp48 원칙 — 부호 일관 + 평균 +1.5%p 이상.
#
# ⚠️ GPU 반환은 프로세스 종료가 아니라 여유 메모리로 판정한다. vLLM의 EngineCore 자식이
#    부모보다 오래 살아남아 21GB를 붙들고 있었다(2026-08-14).
# ⚠️ pgrep 패턴의 대괄호는 자기 자신 매칭 방지(8/13 PRM 체인 교착).
#
# 사용: setsid nohup bash remote/run_exp53c_chain.sh > exp53c_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

OUTDIR=outputs/qlora_teacher_full
ADAPTER=$OUTDIR/qlora_r16_lr5e-05_ep1_final

wait_gpu() {
  for i in $(seq 1 120); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "${FREE:-0}" -gt 18000 ] && [ "$APPS" -eq 0 ]; then return 0; fi
    if [ "$i" -gt 3 ] && ! pgrep -f 'dump_sample[s].py' > /dev/null; then
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
        echo "  고아 GPU 프로세스 정리: $p"; kill "$p" 2>/dev/null || true
      done
    fi
    sleep 30
  done
}

echo "[$(date +%H:%M)] 교사 생성 완료 대기 (목표 1200, 최대 4시간)"
for i in $(seq 1 48); do
  N=$(grep -ho '^## ID: .*' data/teacher_out/*.txt 2>/dev/null | sort -u | wc -l)
  [ "$N" -ge 1195 ] && break
  # 감시 루프까지 멎었다면 더 기다려도 늘지 않는다 — 있는 것으로 진행한다
  if ! pgrep -f 'teacher_superviso[r].sh' > /dev/null; then
    echo "  감시 루프 종료됨 — 확보한 $N문제로 진행"; break
  fi
  sleep 300
done
echo "[$(date +%H:%M)] 교사 완료 $(grep -ho '^## ID: .*' data/teacher_out/*.txt | sort -u | wc -l)문제"

echo "[$(date +%H:%M)] 채점 + 학습 데이터 재구성"
uv run python api/parse_desktop_output.py --in-glob 'data/teacher_out/out_*.txt' \
  --gold data/hard_problems_top.csv --tag cc
uv run python api/build_sft_from_teacher.py --teacher data/teacher_cc.jsonl \
  --src data/hard_problems_top.csv --out data/sft_teacher_full.jsonl

wait_gpu
echo "[$(date +%H:%M)] 학습 시작 (1에폭, 전량)"
uv run python remote/train_qlora.py --data-path data/sft_teacher_full.jsonl \
  --output-dir "$OUTDIR" --epochs 1 > train53c.log 2>&1

if [ ! -d "$ADAPTER" ]; then
  echo "⛔ 어댑터 없음: $ADAPTER — 학습 실패. 중단"; tail -20 train53c.log; exit 1
fi

for S in 42 43 44; do
  OUT=results/val_samples_teacher_full_s$S.jsonl
  [ -s "$OUT" ] && { echo "[$(date +%H:%M)] seed $S 이미 있음"; continue; }
  wait_gpu
  echo "[$(date +%H:%M)] seed $S 덤프"
  uv run python remote/dump_samples.py --n 32 --seed "$S" --adapter "$ADAPTER" --out "$OUT"
done

echo
echo "===== H23 최종 판정: 시드 대응 비교 ====="
for S in 42 43 44; do
  case $S in 42) BASE=results/val_samples.jsonl ;; *) BASE=results/val_samples_s$S.jsonl ;; esac
  echo "--- seed $S 베이스"
  uv run python remote/analyze_selection_gap.py --samples "$BASE" --n 32 | sed -n '2,3p;7p'
  echo "--- seed $S 교사(전량,1에폭)"
  uv run python remote/analyze_selection_gap.py --samples "results/val_samples_teacher_full_s$S.jsonl" --n 32 | sed -n '2,3p;7p'
done
echo "[$(date +%H:%M)] EXP53C_DONE"
echo "판정: 3시드 부호 일관 + 평균 +1.5%p 이상일 때만 채택. 최종 스택 변경은 사용자 승인 사항."
