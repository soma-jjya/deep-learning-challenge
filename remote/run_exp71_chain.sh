#!/bin/bash
# exp71 — 새로 드러난 패턴(pass@32를 지키면 LB에서 이긴다)에 따른 다음 후보 생성.
#
# 2026-08-16 관측: 로컬에서 진 변형 셋이 전부 LB에서 기존 최고를 넘었고, 넷 중 pass@32가
# 떨어진 하나만 LB에서도 떨어졌다. 우리 로컬 gold가 어려운 문항에서 깨져 있다는
# exp52b와 맞물리는 기제다.
#
# 그래서 이번 후보는 전부 **pass@n을 지키거나 올리는 방향**이다:
#   1) 혼합 투표(베이스16 + 교사16) — 합집합 pass@64가 90.5%로 우리가 가진 최고 (GPU 0)
#   2) 교사 어댑터 @ temp 1.1        — 두 기제를 겹친다
#   3) 교사 어댑터 @ seed 43         — 0.78820이 실력인지 운인지 가르는 재실행 변동 측정
#
# ⚠️ 3번이 없으면 1·2번 해석이 불가능하다. 재실행 변동이 0.48%p인데 교사증류의 이득이
#    +0.361%p이므로, 같은 설정 다른 시드가 얼마나 흔들리는지 모르면 아무 말도 못 한다.
#
# 사용: setsid nohup bash remote/run_exp71_chain.sh > exp71_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

TEACHER=outputs/qlora_teacher_full/qlora_r16_lr5e-05_ep1_final

wait_gpu() {
  for i in $(seq 1 240); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "${FREE:-0}" -gt 18000 ] && [ "$APPS" -eq 0 ]; then return 0; fi
    sleep 30
  done
}

# ── 1) GPU 없이 되는 것부터 ──
echo "[$(date +%H:%M)] 혼합 투표 제출 파일 (GPU 0)"
for R in "16 16 lb_mix1616" "24 8 lb_mix248" "8 24 lb_mix824"; do
  set -- $R
  uv run python remote/make_mixed_submission.py \
    --a results/lb_samples.jsonl --b results/lb_samples_teacher.jsonl \
    --ka "$1" --kb "$2" --tag "$3"
done

# ── 2) 교사 어댑터 @ temp 1.1 ──
D=results/lb_samples_teacher_t11.jsonl
if [ ! -s "$D" ] || [ "$(wc -l < "$D")" -lt 831 ]; then
  wait_gpu
  echo "[$(date +%H:%M)] 교사 어댑터 @ temp 1.1 덤프"
  uv run python remote/dump_lb_samples.py --n 32 --seed 42 --temp 1.1 \
    --adapter "$TEACHER" --out "$D"
fi
uv run python remote/make_submission_from_dump.py --dump "$D" --rule weighted --n 32 --tag lb_teacher_t11

# ── 3) 교사 어댑터 @ seed 43 — 재실행 변동 측정 ──
D2=results/lb_samples_teacher_s43.jsonl
if [ ! -s "$D2" ] || [ "$(wc -l < "$D2")" -lt 831 ]; then
  wait_gpu
  echo "[$(date +%H:%M)] 교사 어댑터 @ seed 43 덤프"
  uv run python remote/dump_lb_samples.py --n 32 --seed 43 \
    --adapter "$TEACHER" --out "$D2"
fi
uv run python remote/make_submission_from_dump.py --dump "$D2" --rule weighted --n 32 --tag lb_teacher_s43

echo
echo "[$(date +%H:%M)] EXP71_DONE"
ls -l results/submission_lb_mix*.csv results/submission_lb_teacher_t11.csv results/submission_lb_teacher_s43.csv 2>/dev/null
