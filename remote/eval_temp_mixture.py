"""온도 혼합 앙상블 — 서로 다른 temperature의 표본을 한 표결에 합친다.

발상: 단일 온도로 n을 키우면 같은 다양성 영역에서 표만 늘어 정체(n8~64 실측).
      낮은 온도(정밀·수렴) + 높은 온도(탐색·다양)를 섞으면 표본 분포 자체가 달라진다.
      단순 n 확대와 달리 '다른 종류의 표'를 얻는 것이 목적.

사용:
    uv run python remote/eval_temp_mixture.py --spec 0.4:8,0.7:8,1.0:8
"""
import argparse
import json
import math
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


def weighted_pick(pairs):
    """(답, 평균 로그확률) 목록에서 확신도 가중 다수결."""
    pairs = [(a, lp) for a, lp in pairs if a is not None]
    if not pairs:
        return 0
    m = max(lp for _, lp in pairs)
    w = {}
    for a, lp in pairs:
        w[a] = w.get(a, 0.0) + math.exp((lp - m) * 2.0)
    return max(w, key=w.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default='0.4:8,0.7:8,1.0:8',
                    help='온도:샘플수 목록 (쉼표 구분)')
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=2048)
    args = ap.parse_args()

    parts = []
    for chunk in args.spec.split(','):
        t, n = chunk.split(':')
        parts.append((float(t), int(n)))
    total_n = sum(n for _, n in parts)
    print(f'온도 혼합: {parts} (총 {total_n}표)')

    from vllm import LLM, SamplingParams

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        bad_ids = set(pd.read_csv(bad)['id'])
        val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    print(f'검증 {len(val)}문항')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in val['question']]

    # 온도별로 생성해 문제별 표를 누적
    pooled = [[] for _ in range(len(val))]
    per_temp = {}
    for t, n in parts:
        outs = llm.generate(prompts, SamplingParams(
            n=n, temperature=t, top_p=args.top_p, max_tokens=args.max_tokens,
            seed=42, logprobs=0))
        hit = 0
        for i, o in enumerate(outs):
            votes = []
            for c in o.outputs:
                a = extract_answer(c.text)
                lp = c.cumulative_logprob / max(1, len(c.token_ids))
                pooled[i].append((a, lp))
                votes.append(a)
            v = [x for x in votes if x is not None]
            if v and Counter(v).most_common(1)[0][0] == int(val['answer'][i]):
                hit += 1
        per_temp[f't{t}_n{n}'] = hit / len(val)
        print(f'  [단독 t={t} n={n}] 다수결 {hit/len(val):.1%}')

    maj = sum(1 for i in range(len(val))
              if (lambda v: v and Counter(v).most_common(1)[0][0] == int(val['answer'][i]))(
                  [a for a, _ in pooled[i] if a is not None]))
    wgt = sum(1 for i in range(len(val))
              if weighted_pick(pooled[i]) == int(val['answer'][i]))
    n = len(val)
    print(f'[혼합 {total_n}표 다수결] {maj/n:.1%} ({maj}/{n})')
    print(f'[혼합 {total_n}표 가중] {wgt/n:.1%} ({wgt}/{n})')

    os.makedirs('results', exist_ok=True)
    res = {'spec': args.spec, 'per_temp': per_temp,
           'mixture_majority': maj / n, 'mixture_weighted': wgt / n}
    json.dump(res, open('results/eval_temp_mixture.json', 'w'), indent=2)
    print('저장: results/eval_temp_mixture.json')


if __name__ == '__main__':
    main()
