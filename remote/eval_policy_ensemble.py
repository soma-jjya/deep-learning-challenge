"""정책 앙상블 평가 (H8) — 서로 다른 정책(베이스/QLoRA/GRPO)이 생성한 풀이를 합동 다수결.

베이스 n=8 + QLoRA 어댑터 n=8 = 16표 다수결, 베이스 n=8 + GRPO 어댑터 n=8 = 16표 다수결.
균일 n=16(단일 정책, exp17 75.6%)과 비교해 정책 다양성 자체의 효과를 검증.

사용: uv run python remote/eval_policy_ensemble.py
"""
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')

QLORA_ADAPTER = 'outputs/qlora/qlora_r16_lr0.0002_ep2_final'
GRPO_ADAPTER = 'outputs/grpo_pilot/grpo_pilot_final'


def majority_vote(answers):
    votes = [a for a in answers if a is not None]
    return Counter(votes).most_common(1)[0][0] if votes else 0


def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        bad_ids = set(pd.read_csv(bad)['id'])
        val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    print(f'검증 {len(val)}문항')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096,
              enable_lora=True, max_lora_rank=64)
    tok = llm.get_tokenizer()
    qlora = LoRARequest('qlora', 1, QLORA_ADAPTER)
    grpo = LoRARequest('grpo', 2, GRPO_ADAPTER)

    prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in val['question']]

    sp = SamplingParams(n=8, temperature=0.7, top_p=0.8, max_tokens=2048, seed=42)

    print('베이스 n=8 생성...')
    base_outs = llm.generate(prompts, sp)
    print('QLoRA 어댑터 n=8 생성...')
    qlora_outs = llm.generate(prompts, sp, lora_request=qlora)
    print('GRPO 어댑터 n=8 생성...')
    grpo_outs = llm.generate(prompts, sp, lora_request=grpo)

    base_answers = [[extract_answer(c.text) for c in o.outputs] for o in base_outs]
    qlora_answers = [[extract_answer(c.text) for c in o.outputs] for o in qlora_outs]
    grpo_answers = [[extract_answer(c.text) for c in o.outputs] for o in grpo_outs]

    gold = [int(a) for a in val['answer']]
    n = len(val)

    def score(combined_per_problem):
        preds = [majority_vote(a) for a in combined_per_problem]
        correct = sum(int(p) == g for p, g in zip(preds, gold))
        return correct / n, correct

    qlora_ens_combined = [b + q for b, q in zip(base_answers, qlora_answers)]
    grpo_ens_combined = [b + g for b, g in zip(base_answers, grpo_answers)]

    qlora_acc, qlora_correct = score(qlora_ens_combined)
    grpo_acc, grpo_correct = score(grpo_ens_combined)

    print(f'[base8+qlora8 ensemble n16] {qlora_acc:.1%} ({qlora_correct}/{n})')
    print(f'[base8+grpo8 ensemble n16] {grpo_acc:.1%} ({grpo_correct}/{n})')

    out = {
        'val_n': n,
        'qlora_ensemble_n16': qlora_acc,
        'qlora_ensemble_n16_correct': qlora_correct,
        'grpo_ensemble_n16': grpo_acc,
        'grpo_ensemble_n16_correct': grpo_correct,
    }
    os.makedirs('results', exist_ok=True)
    json.dump(out, open('results/eval_policy_ensemble.json', 'w'), indent=2)
    print('저장: results/eval_policy_ensemble.json')


if __name__ == '__main__':
    main()
