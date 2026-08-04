"""RFT 학습 데이터 생성 — AWS GPU 서버에서 실행 (vLLM).

train 문제마다 베이스 모델로 풀이를 N개 샘플링하고, 정답에 도달한 풀이만 골라
QLoRA SFT용 데이터(data/sft.jsonl)를 만든다. (RFT: arXiv:2308.01825, STaR)

사용법 (서버 tmux 안에서):
    uv pip install vllm   # 최초 1회
    nohup uv run python remote/generate_rft.py > rft.log 2>&1 &

- 검증 세트(500문제, seed=123 — 노트북 02와 동일)는 자동 제외
- 청크마다 중간 저장 → 스팟 회수·중단 시 재실행하면 이어서 진행
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

CONFIG = dict(
    model='Qwen/Qwen2.5-3B-Instruct',
    train_csv='deep-learning-challenge-2026/deep_chal_math_train.csv',
    out_path='data/sft.jsonl',
    progress_path='data/rft_progress.json',
    n_samples=6,          # 문제당 샘플 수 (다양한 풀이 경로가 성능을 좌우 — RFT 논문)
    max_keep=3,           # 문제당 정답 풀이 최대 채택 수 (중복 학습 방지)
    temperature=0.8,
    top_p=0.95,
    max_tokens=2048,
    chunk_size=500,       # 이 단위로 중간 저장
    val_n=500,
    val_seed=123,         # 노트북 02와 반드시 동일
)

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.'
)


def main():
    from vllm import LLM, SamplingParams

    df = pd.read_csv(CONFIG['train_csv'])
    df.columns = df.columns.str.strip()
    val_ids = set(df.sample(CONFIG['val_n'], random_state=CONFIG['val_seed'])['id'])
    df = df[~df['id'].isin(val_ids)].reset_index(drop=True)
    print(f'대상 문제 {len(df)}개 (검증 {len(val_ids)}개 제외)')

    os.makedirs('data', exist_ok=True)
    done_ids = set()
    if os.path.exists(CONFIG['progress_path']):
        done_ids = set(json.load(open(CONFIG['progress_path']))['done_ids'])
        print(f'이어서 진행: {len(done_ids)}개 완료됨')
    todo = df[~df['id'].isin(done_ids)].reset_index(drop=True)

    llm = LLM(model=CONFIG['model'], dtype='bfloat16', gpu_memory_utilization=0.92)
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(n=CONFIG['n_samples'], temperature=CONFIG['temperature'],
                        top_p=CONFIG['top_p'], max_tokens=CONFIG['max_tokens'], seed=42)

    def to_prompt(q):
        return tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': q}],
            tokenize=False, add_generation_prompt=True)

    kept_total = solved = 0
    for start in range(0, len(todo), CONFIG['chunk_size']):
        chunk = todo.iloc[start:start + CONFIG['chunk_size']]
        outs = llm.generate([to_prompt(q) for q in chunk['question']], sp)
        with open(CONFIG['out_path'], 'a', encoding='utf-8') as f:
            for row, out in zip(chunk.itertuples(), outs):
                gold = int(row.answer)
                kept = []
                for comp in out.outputs:
                    text = comp.text
                    if extract_answer(text) == gold and len(kept) < CONFIG['max_keep']:
                        # 서로 너무 비슷한 풀이는 앞 200자 비교로 대충 걸러낸다
                        if not any(text[:200] == k[:200] for k in kept):
                            kept.append(text)
                if kept:
                    solved += 1
                for text in kept:
                    f.write(json.dumps({'id': row.id, 'messages': [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': row.question},
                        {'role': 'assistant', 'content': text.strip()},
                    ]}, ensure_ascii=False) + chr(10))
                kept_total += len(kept)
                done_ids.add(row.id)
        json.dump({'done_ids': sorted(done_ids)}, open(CONFIG['progress_path'], 'w'))
        print(f'진행 {len(done_ids)}/{len(df)} | 정답 도달 문제 {solved} | 채택 풀이 {kept_total}')

    print(f'완료: {CONFIG["out_path"]} — 채택 풀이 {kept_total}개')
    print('다음: remote/train_qlora.py 로 SFT 학습')


if __name__ == '__main__':
    main()
