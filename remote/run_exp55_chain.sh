#!/bin/bash
# exp55 — max_tokens 2048 → 3072. 잘림 자체를 줄여 표를 회수한다.
#
# 근거(analyze_truncation.py, GPU 없이 기존 덤프로 측정):
#   - 잘린 표는 전체의 1.7%뿐이지만 **오답 문항에 12배 몰려 있다**
#     (정답 문항 문항당 0.15표 vs 오답 문항 1.80표)
#   - "틀렸고 · 잘린 표가 있고 · 안 잘린 표 어디에도 정답이 없는" 표적 집합이
#     시드 42/43/44에서 각각 28 / 30 / 28문항 = **5.8~6.2%p**
#   - 이 중 20~25%만 회수해도 +1.2~1.5%p로 2등과의 격차와 같은 크기다
#
# 왜 지금까지 안 했나: 잘림을 **집계 규칙**으로만 다뤄왔다(drop_trunc = 잘린 표 버리기).
# 그건 "이미 버려진 표를 어떻게 셀까"의 문제이고, 잘림 자체를 줄이는 축은
# H1(1024→2048) 이후 한 번도 재검토하지 않았다.
#
# 판정: exp48 원칙 — 시드 42/43/44 부호 일관 + 평균 +1.5%p 이상.
# 부수 측정: 생성 시간. 8/31 최종전은 시간 제약이 있으므로 느려진 만큼을 반드시 기록한다.
#
# ⚠️ pgrep 패턴의 대괄호는 자기 자신 매칭 방지(8/13 PRM 체인 교착).
#
# 사용: setsid nohup bash remote/run_exp55_chain.sh > exp55_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

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

for S in 42 43 44; do
  OUT=results/val_samples_mt3072_s$S.jsonl
  [ -s "$OUT" ] && { echo "[$(date +%H:%M)] seed $S 이미 있음"; continue; }
  wait_gpu
  echo "[$(date +%H:%M)] seed $S 덤프 (max_tokens=3072)"
  T0=$(date +%s)
  uv run python remote/dump_samples.py --n 32 --seed "$S" --max-tokens 3072 --out "$OUT"
  echo "[$(date +%H:%M)] seed $S 소요 $(( ($(date +%s) - T0) / 60 ))분"
done

echo
echo "===== exp55 판정: 시드 대응 비교 (max_tokens 2048 vs 3072) ====="
for S in 42 43 44; do
  case $S in 42) BASE=results/val_samples.jsonl ;; *) BASE=results/val_samples_s$S.jsonl ;; esac
  echo "--- seed $S : 2048 (기준)"
  uv run python remote/analyze_truncation.py --samples "$BASE" --n 32 | sed -n '3,10p'
  echo "--- seed $S : 3072"
  uv run python remote/analyze_truncation.py --samples "results/val_samples_mt3072_s$S.jsonl" --n 32 | sed -n '3,10p'
done
echo "[$(date +%H:%M)] EXP55_DONE"
echo "판정: 3시드 부호 일관 + 평균 +1.5%p 이상일 때만 채택. 생성 시간 증가분도 함께 볼 것."
echo "최종 스택 변경은 사용자 승인 사항."
