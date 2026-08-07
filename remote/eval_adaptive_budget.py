"""적응형 예산 평가 (exp22). n=8 다수결에서 최다 득표가 3표 이하(합의 실패)인 문제만
추가 24표(합계 32표)로 재표결하고, 나머지는 n=8 결과를 그대로 유지한다.

비교 기준: n8 74.7%(exp05), 균일 n16 75.6%(exp07). 성공 기준: >=76.3%.
사용: uv run python remote/eval_adaptive_budget.py
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
    'Put your final integer answer inside ' + BS + 'boxed{}.'
)

CONSENSUS_THRESHOLD = 3  # 최다 득표 <= 3 이면 합의 실패
EXTRA_N = 24


def majority(answers):
    votes = [a for a in answers if a is not None]
    if not votes:
        return 0, 0
    top, cnt = Counter(votes).most_common(1)[0]
    return top, cnt


def main():
    from vllm import LLM, SamplingParams

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad_path = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad_path):
        bad_ids = set(pd.read_csv(bad_path)['id'])
        val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    print(f'검증 문항 {len(val)}개 (오류 문항 제외 후)')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()

    prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in val['question']]

    print(f'1단계: n=8 생성 ({len(prompts)}문항)')
    outs8 = llm.generate(prompts, SamplingParams(
        n=8, temperature=0.7, top_p=0.8, max_tokens=2048, seed=42))
    answers8 = [[extract_answer(c.text) for c in o.outputs] for o in outs8]

    n8_top = []
    consensus_fail_idx = []
    for i, answers in enumerate(answers8):
        top, cnt = majority(answers)
        n8_top.append(top)
        if cnt <= CONSENSUS_THRESHOLD:
            consensus_fail_idx.append(i)

    n8_acc = sum(int(t) == int(a) for t, a in zip(n8_top, val['answer'])) / len(val)
    print(f'[n8 baseline] 정확도: {n8_acc:.1%}')
    print(f'합의 실패 문제 수: {len(consensus_fail_idx)} / {len(val)} '
          f'({len(consensus_fail_idx) / len(val):.1%})')

    all_answers = [list(a) for a in answers8]  # 문제별 답 리스트 (초기 8개)

    if consensus_fail_idx:
        extra_prompts = [prompts[i] for i in consensus_fail_idx]
        print(f'2단계: 합의 실패 {len(extra_prompts)}문항에 추가 n={EXTRA_N} 생성')
        outs_extra = llm.generate(extra_prompts, SamplingParams(
            n=EXTRA_N, temperature=0.7, top_p=0.8, max_tokens=2048, seed=43))
        for idx, o in zip(consensus_fail_idx, outs_extra):
            all_answers[idx].extend(extract_answer(c.text) for c in o.outputs)

    adaptive_top = []
    for i, answers in enumerate(all_answers):
        top, _ = majority(answers)
        adaptive_top.append(top)

    adaptive_acc = sum(int(t) == int(a) for t, a in zip(adaptive_top, val['answer'])) / len(val)
    print(f'[adaptive n8/n32] 정확도: {adaptive_acc:.1%}')

    n_extra_votes_used = len(consensus_fail_idx) * (8 + EXTRA_N)
    n_saved_votes = (len(val) - len(consensus_fail_idx)) * 8 + n_extra_votes_used
    avg_votes_per_problem = sum(len(a) for a in all_answers) / len(val)
    print(f'문제당 평균 투표 수: {avg_votes_per_problem:.2f} (균일 n16이면 16.00)')

    results = {
        'val_n': len(val),
        'n8_acc': n8_acc,
        'consensus_fail_count': len(consensus_fail_idx),
        'consensus_fail_ratio': len(consensus_fail_idx) / len(val),
        'adaptive_acc': adaptive_acc,
        'avg_votes_per_problem': avg_votes_per_problem,
    }
    os.makedirs('results', exist_ok=True)
    json.dump(results, open('results/eval_adaptive_budget.json', 'w'), indent=2)
    print('저장: results/eval_adaptive_budget.json')


if __name__ == '__main__':
    main()
