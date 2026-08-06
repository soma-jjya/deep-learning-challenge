"""검증자 학습 데이터 생성 (H9) — (문제, 풀이, 정답여부) 라벨 쌍.

베이스 모델로 문제당 4개 풀이를 샘플링하고, 답 일치 여부로 자동 라벨링.
사용: nohup uv run python remote/gen_verifier_data.py > gen_verifier.log 2>&1 &
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

CONFIG = dict(
    n_problems=6000, n_samples=4, temperature=0.8, top_p=0.95,
    max_tokens=2048, chunk=500, out='data/verifier.jsonl',
    progress='data/verifier_progress.json', seed=42,
)

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
    val_ids = set(df.sample(500, random_state=123)['id'])
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    bad_ids = set(pd.read_csv(bad)['id']) if os.path.exists(bad) else set()
    df = df[~df['id'].isin(val_ids | bad_ids)].sample(
        CONFIG['n_problems'], random_state=CONFIG['seed']).reset_index(drop=True)

    os.makedirs('data', exist_ok=True)
    done = set()
    if os.path.exists(CONFIG['progress']):
        done = set(json.load(open(CONFIG['progress']))['done'])
    todo = df[~df['id'].isin(done)].reset_index(drop=True)
    print(f'대상 {len(df)}문제, 남은 {len(todo)}문제')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.92)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=CONFIG['n_samples'], temperature=CONFIG['temperature'],
                        top_p=CONFIG['top_p'], max_tokens=CONFIG['max_tokens'], seed=7)

    pos = neg = 0
    for s in range(0, len(todo), CONFIG['chunk']):
        chunk = todo.iloc[s:s + CONFIG['chunk']]
        prompts = [tok.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': q}],
            tokenize=False, add_generation_prompt=True) for q in chunk['question']]
        outs = llm.generate(prompts, sp)
        with open(CONFIG['out'], 'a', encoding='utf-8') as f:
            for row, out in zip(chunk.itertuples(), outs):
                gold = int(row.answer)
                for c in out.outputs:
                    ans = extract_answer(c.text)
                    label = ans is not None and ans == gold
                    pos += label
                    neg += (not label)
                    f.write(json.dumps({'id': row.id, 'question': row.question,
                                        'solution': c.text.strip()[:6000],
                                        'label': bool(label)}, ensure_ascii=False) + chr(10))
                done.add(row.id)
        json.dump({'done': sorted(done)}, open(CONFIG['progress'], 'w'))
        print(f'진행 {len(done)}/{len(df)} | 정답풀이 {pos} / 오답풀이 {neg}')
    print(f'완료: {CONFIG["out"]} (양성 {pos}, 음성 {neg})')


if __name__ == '__main__':
    main()
