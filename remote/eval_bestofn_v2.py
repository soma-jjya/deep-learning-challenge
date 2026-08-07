"""검증자 v2 기반 Best-of-N 평가 (H9 재도전, exp23c) — 베이스가 생성, v2 어댑터가
근거+Verdict으로 채점(즉답 Yes/No였던 v1과 달리 재검산 근거를 먼저 생성).

세 가지 선택 전략을 같은 표본에서 비교:
  1) majority: 일반 다수결 (기준선)
  2) verdict-vote: 답별로 Verdict=Yes 개수가 최다인 답 선택
  3) hybrid: 답별로 (득표수 + Yes개수) 합산이 최대인 답 선택
사용: uv run python remote/eval_bestofn_v2.py --verifier outputs/verifier_v2/verifier_final --n 8
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')
VERIFY_PROMPT = (
    'You are a strict math solution grader. Re-derive the key steps of the proposed '
    'solution to check whether its final answer is correct. Be brief (a few lines). '
    'End with exactly one line: "Verdict: Yes" if the final answer is correct, '
    'or "Verdict: No" if it is not.')


def verdict_of(text):
    t = text.lower()
    i = t.rfind('verdict:')
    if i < 0:
        return None
    tail = t[i + 8:i + 20]
    if 'yes' in tail:
        return True
    if 'no' in tail:
        return False
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verifier', required=True)
    ap.add_argument('--n', type=int, default=8)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        bad_ids = set(pd.read_csv(bad)['id'])
        val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    print(f'검증 {len(val)}문항, n={args.n}')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096,
              enable_lora=True, max_lora_rank=64)
    tok = llm.get_tokenizer()
    verifier = LoRARequest('verifier_v2', 1, args.verifier)

    # 1단계: 베이스(무어댑터)로 n개 풀이 생성
    gen_prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in val['question']]
    outs = llm.generate(gen_prompts, SamplingParams(
        n=args.n, temperature=0.7, top_p=0.8, max_tokens=2048, seed=42))

    # 2단계: v2 검증자 어댑터로 각 풀이를 근거+Verdict으로 채점 (greedy)
    flat_prompts, meta = [], []
    for i, o in enumerate(outs):
        for j, c in enumerate(o.outputs):
            flat_prompts.append(tok.apply_chat_template(
                [{'role': 'system', 'content': VERIFY_PROMPT},
                 {'role': 'user', 'content': 'Problem:' + chr(10) + val['question'][i] +
                  chr(10) * 2 + 'Proposed solution:' + chr(10) + c.text.strip()[:4000]}],
                tokenize=False, add_generation_prompt=True))
            meta.append((i, j))
    judges = llm.generate(flat_prompts, SamplingParams(
        temperature=0, max_tokens=600), lora_request=verifier)

    verdict_map = defaultdict(dict)
    for (i, j), s in zip(meta, judges):
        verdict_map[i][j] = verdict_of(s.outputs[0].text)

    res = {'majority': 0, 'verdict_vote': 0, 'hybrid': 0}
    for i, o in enumerate(outs):
        answers = [extract_answer(c.text) for c in o.outputs]
        gold = int(val['answer'][i])
        votes = [a for a in answers if a is not None]
        maj = Counter(votes).most_common(1)[0][0] if votes else 0

        vote_count = defaultdict(int)
        yes_count = defaultdict(int)
        for j, a in enumerate(answers):
            if a is None:
                continue
            vote_count[a] += 1
            if verdict_map[i].get(j):
                yes_count[a] += 1

        vv = max(yes_count, key=yes_count.get) if any(yes_count.values()) else maj
        hybrid_score = {a: vote_count[a] + yes_count[a] for a in vote_count}
        hy = max(hybrid_score, key=hybrid_score.get) if hybrid_score else maj

        res['majority'] += int(maj == gold)
        res['verdict_vote'] += int(vv == gold)
        res['hybrid'] += int(hy == gold)

    n = len(val)
    out = {k: v / n for k, v in res.items()}
    for k, v in out.items():
        print(f'[{k}] {v:.1%} ({res[k]}/{n})')
    os.makedirs('results', exist_ok=True)
    json.dump(out, open('results/eval_bestofn_v2.json', 'w'), indent=2)
    print('저장: results/eval_bestofn_v2.json')


if __name__ == '__main__':
    main()
