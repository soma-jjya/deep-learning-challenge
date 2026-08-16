#!/bin/bash
# 생성 완료를 기다렸다가 implicit PRM 채점을 이어서 실행.
# 명령줄에 "remote/" 가 포함돼 워치독의 작업 판정 패턴에 걸리므로 인스턴스가 꺼지지 않는다.
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge
GEN=results/rm_gen_seed42_n32.jsonl
while pgrep -f "remote/eval_reward_bestofn.py" > /dev/null; do sleep 30; done
if [ ! -f "$GEN" ]; then echo "PRM_ABORT: 생성물 없음"; exit 1; fi
echo "PRM_CHAIN: 생성 완료 ($(wc -l < $GEN)문항) — implicit PRM 채점 시작"
uv run python remote/eval_implicit_prm.py --dpo outputs/dpo/dpo_r8_b0.3_lr5e-06_final --gen "$GEN"
echo "PRM_CHAIN_DONE"
