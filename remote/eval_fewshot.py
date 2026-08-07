"""4-shot 프롬프트 평가 (exp20) — 유형별(기하/대수/정수론/응용) 짧은 예시 풀이 4개를
모든 문제 앞에 예시 대화(user->assistant)로 붙여 greedy와 SC n=8 평가.

예시는 data/sft_short.jsonl에서 선택 (학습에 쓰인 적 없는 순수 few-shot 컨텍스트 용도,
검증 세트(val, seed=123)와는 무관한 train 문제).
비교 기준: zero-shot greedy 69.4% / SC n8 74.7% (exp05).
사용: uv run python remote/eval_fewshot.py
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

# 유형별 짧은 정답 풀이 4개 (data/sft_short.jsonl에서 선정: 기하/대수/정수론/응용 각 1)
FEWSHOT_IDS = ['train-005807', 'train-005399', 'train-002653', 'train-013800']


def load_fewshot_examples():
    recs = {}
    with open('data/sft_short.jsonl') as f:
        for line in f:
            r = json.loads(line)
            if r['id'] in FEWSHOT_IDS:
                recs[r['id']] = r
    examples = []
    for i in FEWSHOT_IDS:
        r = recs[i]
        q = r['messages'][1]['content']
        a = r['messages'][2]['content']
        examples.append((q, a))
    assert len(examples) == 4, f'예시 {len(examples)}개만 찾음 (4개 필요)'
    return examples


def main():
    from vllm import LLM, SamplingParams

    examples = load_fewshot_examples()
    print(f'few-shot 예시 {len(examples)}개 로드 완료')
    for q, a in examples:
        print(' -', q[:70])

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        bad_ids = set(pd.read_csv(bad)['id'])
        val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    print(f'검증 {len(val)}문항 (오류 문항 제외 후)')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.9, max_model_len=4096)
    tok = llm.get_tokenizer()

    def build_messages(q):
        msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        for eq, ea in examples:
            msgs.append({'role': 'user', 'content': eq})
            msgs.append({'role': 'assistant', 'content': ea})
        msgs.append({'role': 'user', 'content': q})
        return msgs

    prompts = [tok.apply_chat_template(build_messages(q), tokenize=False,
                                        add_generation_prompt=True) for q in val['question']]

    results = {'fewshot_ids': FEWSHOT_IDS, 'val_n': len(val)}

    def run(name, sp):
        outs = llm.generate(prompts, sp)
        preds = [Counter(a for c in o.outputs
                          for a in [extract_answer(c.text)] if a is not None).most_common(1)
                  for o in outs]
        preds = [p[0][0] if p else 0 for p in preds]
        acc = sum(int(p) == int(a) for p, a in zip(preds, val['answer'])) / len(val)
        correct = sum(int(p) == int(a) for p, a in zip(preds, val['answer']))
        print(f'[{name}] 정확도: {acc:.1%} ({correct}/{len(val)})')
        results[name] = acc
        return preds

    run('fewshot_greedy', SamplingParams(temperature=0, max_tokens=2048))
    run('fewshot_sc_n8', SamplingParams(n=8, temperature=0.7, top_p=0.8,
                                         max_tokens=2048, seed=42))

    os.makedirs('results', exist_ok=True)
    json.dump(results, open('results/eval_fewshot.json', 'w'), indent=2)
    print('저장: results/eval_fewshot.json')


if __name__ == '__main__':
    main()
