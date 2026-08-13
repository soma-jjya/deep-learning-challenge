#!/bin/bash
# exp48 — DPO 효과가 노이즈인지 실효인지 판정하기 위한 시드 대응 비교.
# 같은 시드에서 베이스와 DPO를 짝지어 뽑아야 표본 변동을 상쇄할 수 있다.
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge
A=outputs/dpo/dpo_r8_b0.3_lr5e-06_final
for S in 43 44; do
  uv run python remote/dump_samples.py --n 32 --seed $S --out results/val_samples_s$S.jsonl
  uv run python remote/dump_samples.py --n 32 --seed $S --adapter $A --out results/val_samples_dpo_s$S.jsonl
done
echo "PAIRED_DUMPS_DONE"
