#!/bin/bash
# exp61 — H28 Ranked Voting Self-Consistency (arXiv 2505.10772).
#
# 왜 이것이 기각 목록과 다른가: 우리가 소진한 집계 규칙 63종은 전부 **투표지는 그대로 두고
# 개표 방식만** 바꾼 것이었다. 이건 **투표지 자체**를 바꾼다 — 각 CoT가 순위 리스트를 낸다.
# 우리 실측과 맞물린다: 다수결 1위가 틀린 119문항에서 정답이 2위 18.5% + 3위 12.6% = 31%.
# 단일답 투표지에서는 이 표들이 애초에 기록되지 않는다.
#
# 논문 평가 체크포인트가 Qwen2.5-3B-Instruct로 우리와 동일하며, 6개 벤치마크 전부에서 SC가
# 최하위였다(평균 SC 60.13 → MRR 65.08). 자유형식 출력 과제에서 최대 이득(+12.73pp)이 나와
# 선택지 집합 없이도 작동한다는 직접 증거가 있다.
#
# ⚠️ 순응률을 먼저 본다. 논문은 퓨샷으로 형식을 가르치지만 우리는 퓨샷이 두 번 다 해로웠으므로
#    제로샷 지시를 쓴다. 2개 이상 순위를 낸 표본이 절반 미만이면 이 실험은 무효이며,
#    그 수치로 가설을 기각해서는 안 된다(exp37이 버그로 0/65건을 호출해놓고 '기각'될 뻔한 사고).
#
# 사용: setsid nohup bash remote/run_exp61_chain.sh > exp61_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

wait_gpu() {
  for i in $(seq 1 480); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "${FREE:-0}" -gt 18000 ] && [ "$APPS" -eq 0 ]; then return 0; fi
    sleep 30
  done
}

echo "[$(date +%H:%M)] 앞선 작업(exp60) 종료 대기"
while pgrep -f 'dump_sample[s].py|train_qlor[a].py' > /dev/null; do sleep 60; done
wait_gpu

# 먼저 시드 42 하나로 순응률과 신호를 본다. 형식을 안 따르면 3시드는 낭비다.
echo "[$(date +%H:%M)] seed 42 생성 + 채점"
uv run python remote/eval_ranked_voting.py --n 32 --seed 42

echo "[$(date +%H:%M)] EXP61_SEED42_DONE"
echo "판정: 순응률(2개 이상 순위)이 50% 이상이고 같은 실행의 top1_weighted 대비"
echo "      +5문항 이상이면 시드 43·44로 확장. 최종 스택 변경은 사용자 승인 사항."
