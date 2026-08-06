"""오답 정밀 분석 (exp16, H13 사전 단계) — SC n=8 평가를 재실행하며 오답 상세 기록.

eval_vllm.py --mode sc --n 8 과 동일한 설정(베이스 모델, temp0.7/top_p0.8/seed42)으로
재현하되, 오답 문제마다 {id, question 앞 200자, 정답, 8개 샘플 추출 답 목록}을
results/wrong_analysis.jsonl에 저장한다.

사용: uv run python remote/eval_error_analysis.py
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


def majority_vote(answers):
    votes = [a for a in answers if a is not None]
    return Counter(votes).most_common(1)[0][0] if votes else 0


def main():
    from vllm import LLM, SamplingParams

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        bad_ids = set(pd.read_csv(bad)['id'])
        val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    print(f'검증 {len(val)}문항 — SC n=8 재실행, 오답 상세 기록')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()

    prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in val['question']]

    sp = SamplingParams(n=8, temperature=0.7, top_p=0.8, max_tokens=2048, seed=42)
    outs = llm.generate(prompts, sp)

    wrong_records = []
    correct = 0
    for i, o in enumerate(outs):
        answers = [extract_answer(c.text) for c in o.outputs]
        pred = majority_vote(answers)
        gold = int(val['answer'][i])
        if int(pred) == gold:
            correct += 1
        else:
            wrong_records.append({
                'id': val['id'][i],
                'question_head': val['question'][i][:200],
                'answer': gold,
                'extracted_answers': answers,
            })

    acc = correct / len(val)
    print(f'[sc_n8] 정확도: {acc:.1%} ({correct}/{len(val)}), 오답 {len(wrong_records)}건')

    os.makedirs('results', exist_ok=True)
    with open('results/wrong_analysis.jsonl', 'w') as f:
        for rec in wrong_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print('저장: results/wrong_analysis.jsonl')

    json.dump({'val_n': len(val), 'sc_n8': acc, 'wrong_count': len(wrong_records)},
              open('results/eval_error_analysis.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
