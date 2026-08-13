#!/bin/bash
# 학습 종료를 기다렸다가 곧바로 평가를 실행하는 체인.
# 명령줄에 "uv run python remote/" 문자열이 포함되므로 러너의 pgrep 검사에 걸려
# 대기 상태가 유지된다 = 러너가 exp47을 중복 학습하지 않는다.
export PATH=$HOME/.local/bin:$PATH
set -a; . ~/.ajudl_env; set +a
cd ~/work/deep-learning-challenge
while pgrep -f "remote/train_dpo.py" > /dev/null; do sleep 30; done
D=outputs/dpo/dpo_r8_b0.3_lr5e-06_final
if [ ! -d "$D" ]; then echo "CHAIN_ABORT: 어댑터 없음 ($D) — 학습 실패로 판단"; exit 1; fi
echo "CHAIN: 학습 완료 확인, 평가 시작 $D"
uv run python remote/eval_vllm.py --mode both --n 32 --adapter "$D" --tag dpo
