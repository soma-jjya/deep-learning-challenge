#!/bin/bash
# exp70 — 로컬에서 졌다는 이유로 제출하지 않은 후보들을 리더보드로 판정한다.
#
# 왜: 우리 정책은 "로컬은 오답의 23%가 라벨 의심인 흐린 자, 리더보드 831문항은 운영진
# 정밀 검수를 거친 깨끗한 자 — 더 좋은 측정기를 아낄 이유가 없다"이다. 그런데 실제로는
# 로컬이 지면 제출을 안 해왔다. 두 자가 어긋난 사례가 이미 둘이다:
#   · exp41: 로컬이 좋아 보였는데 LB가 정반대
#   · exp64 DPO: 로컬 −0.14%p인데 LB +0.12%p (부호 반대)
#
# 그리고 **기제까지 있다**. exp52b에서 우리 검증셋의 **어려운 문항일수록 gold가 깨져 있음**을
# 확인했다(이중조건 표본 30건 중 결함 의심 92.3%, 3건 직접 검산). 그러면 **어려운 문제를
# 겨냥한 변형은 로컬에서 부당하게 벌점을 받고 LB에서는 제대로 평가받는다.**
# 교사증류 어댑터가 정확히 그 경우다 — pass@32가 +1.7문항 높았는데 투표에서 졌다.
#
# 우선순위(로컬이 부당했을 가능성 순):
#   1) 교사증류 어댑터  — pass@32 +1.7문항, 어려운 문제 겨냥
#   2) temp 1.1        — pass@32 유지(426 vs 428), 커버리지 안 잃고 표만 흩어짐
#   3) RFT 마스킹 어댑터 — 마스킹 수정판, LB 미검증
#
# 사용: setsid nohup bash remote/run_exp70_chain.sh > exp70_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

wait_gpu() {
  for i in $(seq 1 240); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "${FREE:-0}" -gt 18000 ] && [ "$APPS" -eq 0 ]; then return 0; fi
    sleep 30
  done
}

run_one() {   # $1=태그  $2=덤프경로  $3...=dump_lb_samples 추가 인자
  local TAG="$1"; shift
  local DUMP="$1"; shift
  if [ -s "$DUMP" ] && [ "$(wc -l < "$DUMP")" -ge 831 ]; then
    echo "[$(date +%H:%M)] $TAG 덤프 이미 완료"
  else
    wait_gpu
    echo "[$(date +%H:%M)] $TAG 덤프 시작"
    uv run python remote/dump_lb_samples.py --n 32 --seed 42 --out "$DUMP" "$@"
  fi
  uv run python remote/make_submission_from_dump.py --dump "$DUMP" \
    --rule weighted --n 32 --tag "$TAG"
  echo "[$(date +%H:%M)] $TAG 제출 파일 준비 완료"
}

# 1) 교사증류 어댑터 — 가장 강한 기제
run_one lb_teacher results/lb_samples_teacher.jsonl \
  --adapter outputs/qlora_teacher_full/qlora_r16_lr5e-05_ep1_final

# 2) temp 1.1 — 커버리지 유지형
run_one lb_t11 results/lb_samples_t11.jsonl --temp 1.1

# 3) RFT 마스킹 어댑터
run_one lb_rftmask results/lb_samples_rftmask.jsonl \
  --adapter outputs/qlora_rft_masked/qlora_r16_lr5e-05_ep1_final

echo
echo "[$(date +%H:%M)] EXP70_DONE — 아래 파일을 로컬에서 검증 후 제출"
ls -l results/submission_lb_teacher.csv results/submission_lb_t11.csv results/submission_lb_rftmask.csv 2>/dev/null
