"""문제 텍스트 정화 효과 평가 (exp21).

remote/clean_question.py의 정화 함수를 적용/미적용한 두 버전으로 greedy 평가를 돌려
전체 정확도와 "영향받은 문항 부분집합"에서의 정확도를 비교한다.

사용: uv run python remote/eval_question_clean.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer
from clean_question import clean_question, is_affected

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.'
)


def main():
    from vllm import LLM, SamplingParams

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        bad_ids = set(pd.read_csv(bad)['id'])
        val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)

    affected = val['question'].apply(is_affected)
    print(f'검증 {len(val)}문항, 정화로 영향받은 문항 {affected.sum()}개')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()

    def build_prompts(questions):
        return [tok.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': q}],
            tokenize=False, add_generation_prompt=True) for q in questions]

    orig_prompts = build_prompts(val['question'])
    clean_prompts = build_prompts(val['question'].apply(clean_question))

    sp = SamplingParams(temperature=0, max_tokens=2048)
    orig_outs = llm.generate(orig_prompts, sp)
    clean_outs = llm.generate(clean_prompts, sp)

    orig_preds = [extract_answer(o.outputs[0].text) for o in orig_outs]
    clean_preds = [extract_answer(o.outputs[0].text) for o in clean_outs]

    def acc(preds, mask=None):
        idx = range(len(val)) if mask is None else [i for i in range(len(val)) if mask[i]]
        if not idx:
            return None, 0
        correct = sum(1 for i in idx if preds[i] is not None and int(preds[i]) == int(val['answer'][i]))
        return correct / len(idx), len(idx)

    orig_acc, n_all = acc(orig_preds)
    clean_acc, _ = acc(clean_preds)
    orig_acc_aff, n_aff = acc(orig_preds, affected)
    clean_acc_aff, _ = acc(clean_preds, affected)

    print(f'[전체 {n_all}문항] 원문 greedy: {orig_acc:.1%} / 정화 greedy: {clean_acc:.1%}')
    print(f'[영향받은 {n_aff}문항] 원문 greedy: {orig_acc_aff:.1%} / 정화 greedy: {clean_acc_aff:.1%}')

    results = {
        'val_n': len(val),
        'affected_n': int(affected.sum()),
        'orig_greedy_all': orig_acc,
        'clean_greedy_all': clean_acc,
        'orig_greedy_affected': orig_acc_aff,
        'clean_greedy_affected': clean_acc_aff,
    }
    os.makedirs('results', exist_ok=True)
    json.dump(results, open('results/eval_question_clean.json', 'w'), indent=2)
    print('저장: results/eval_question_clean.json')


if __name__ == '__main__':
    main()
