"""프롬프트 앙상블 SC (H19) — 서로 다른 시스템 프롬프트 4종 × 2샘플 = 8표 다수결.

발상: SC의 힘은 풀이 다양성에서 온다 → 프롬프트로 다양성을 인위적으로 확대.
비교 대상: 단일 프롬프트 SC n=8 (74.7%).
사용: uv run python remote/eval_prompt_ensemble.py
"""
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

BOX = 'Put your final integer answer inside ' + BS + 'boxed{}.'
PROMPTS = [
    'You are an expert competition mathematician. Solve the problem step by step. '
    'The final answer is always an integer. ' + BOX,
    'You are a careful mathematician. Solve the problem, then double-check your '
    'arithmetic before giving the final answer. The final answer is always an integer. ' + BOX,
    'You are a mathematician who prefers algebraic approaches. Set up equations and '
    'solve them systematically. The final answer is always an integer. ' + BOX,
    'You are a pragmatic problem solver. Find the shortest correct path to the answer, '
    'avoiding unnecessary steps. The final answer is always an integer. ' + BOX,
]
N_PER_PROMPT = 2


def main():
    from vllm import LLM, SamplingParams

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        bad_ids = set(pd.read_csv(bad)['id'])
        val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    print(f'검증 {len(val)}문항, 프롬프트 {len(PROMPTS)}종 × {N_PER_PROMPT}샘플')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.9, max_model_len=4096)
    tok = llm.get_tokenizer()

    # 문제 × 프롬프트 조합을 한 배열로 펼쳐 생성
    flat = []
    for q in val['question']:
        for sp_text in PROMPTS:
            flat.append(tok.apply_chat_template(
                [{'role': 'system', 'content': sp_text},
                 {'role': 'user', 'content': q}],
                tokenize=False, add_generation_prompt=True))
    sp = SamplingParams(n=N_PER_PROMPT, temperature=0.7, top_p=0.8,
                        max_tokens=2048, seed=42)
    outs = llm.generate(flat, sp)

    k = len(PROMPTS)
    correct = 0
    for i in range(len(val)):
        votes = []
        for j in range(k):
            for c in outs[i * k + j].outputs:
                a = extract_answer(c.text)
                if a is not None:
                    votes.append(a)
        pred = Counter(votes).most_common(1)[0][0] if votes else 0
        if int(pred) == int(val['answer'][i]):
            correct += 1
    acc = correct / len(val)
    print(f'[prompt-ensemble n8] 정확도: {acc:.1%} ({correct}/{len(val)})')

    os.makedirs('results', exist_ok=True)
    json.dump({'prompt_ensemble_n8': acc}, open('results/eval_prompt_ensemble.json', 'w'), indent=2)
    print('저장: results/eval_prompt_ensemble.json')


if __name__ == '__main__':
    main()
