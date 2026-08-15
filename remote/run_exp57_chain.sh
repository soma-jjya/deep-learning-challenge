#!/bin/bash
# exp57 — H26 검색 기반 퓨샷. 먼저 시드 1개로 신호 유무를 보고, 있으면 그때 확장한다.
#
# 왜 단계적으로: 시드 3개는 GPU 2시간이다. exp20(고정 퓨샷)이 −1.6%p였으므로 사전 확률이
# 높지 않다. 신호가 없으면 한 시드에서 끝내고, 있으면 exp48 원칙대로 3시드로 판정한다.
#
# ⚠️ 로컬 이득을 그대로 믿지 말 것. 검증셋이 train에서 뽑혔으므로 검색 풀에 거의 같은 문제가
# 있을 수 있고(실측: 유사도 0.700짜리 사실상 동일 문항 존재), 실제 test에는 그런 쌍이 적을 수
# 있다. 로컬에서 신호가 나오면 **리더보드로 확인**하는 것이 이 가설의 진짜 판정이다.
#
# 사용: setsid nohup bash remote/run_exp57_chain.sh > exp57_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

wait_gpu() {
  for i in $(seq 1 240); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "${FREE:-0}" -gt 18000 ] && [ "$APPS" -eq 0 ]; then return 0; fi
    if [ "$i" -gt 3 ] && ! pgrep -f 'dump_sample[s].py|eval_retrieval_fewsho[t].py' > /dev/null; then
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
        echo "  고아 GPU 프로세스 정리: $p"; kill "$p" 2>/dev/null || true
      done
    fi
    sleep 30
  done
}

echo "[$(date +%H:%M)] 앞선 작업(exp55) 종료 대기"
while pgrep -f 'dump_sample[s].py' > /dev/null; do sleep 60; done
wait_gpu

echo "[$(date +%H:%M)] k=3, seed 42 생성"
uv run python remote/eval_retrieval_fewshot.py --k 3 --n 32 --seed 42

echo
echo "===== seed 42 : 베이스 vs 검색 퓨샷 ====="
echo "--- 베이스(제로샷)"
uv run python remote/analyze_selection_gap.py --samples results/val_samples.jsonl --n 32 | sed -n '2,7p'
echo "--- 검색 퓨샷 k=3"
uv run python remote/analyze_selection_gap.py --samples results/val_samples_rfs3_s42.jsonl --n 32 | sed -n '2,7p'
echo "[$(date +%H:%M)] EXP57_SEED42_DONE"
echo "신호가 있으면(+5문항 이상) 시드 43·44로 확장, 이어서 리더보드 확인."
