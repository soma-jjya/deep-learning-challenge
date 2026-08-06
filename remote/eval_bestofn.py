"""검증자 기반 Best-of-N 평가 (H9) — 베이스가 생성, 검증자 어댑터가 채점.

세 가지 선택 전략을 같은 표본에서 비교:
  1) majority: 일반 다수결 (기준선)
  2) verifier_weighted: 답별로 검증자 P(Yes) 합산 최대
  3) best_of_1: 검증자 점수 최고 풀이의 답
사용: uv run python remote/eval_bestofn.py --verifier outputs/verifier/verifier_final --n 8
"""
import argparse
import json
import math
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
    'You are a strict math solution grader. You will be given a problem and a proposed '
    'solution. Judge whether the final answer of the solution is correct. '
    'Reply with exactly one word: Yes or No.')


def p_yes(first_token_logprobs, tok):
    """첫 생성 토큰의 top logprobs에서 P(Yes)를 근사."""
    if not first_token_logprobs:
        return 0.0
    yes = no = None
    for tid, lp in first_token_logprobs.items():
        text = tok.decode([tid]).strip().lower()
        if text == 'yes' and yes is None:
            yes = lp.logprob
        elif text == 'no' and no is None:
            no = lp.logprob
    if yes is None:
        return 0.0
    if no is None:
        return 1.0
    ey, en = math.exp(yes), math.exp(no)
    return ey / (ey + en)


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
    verifier = LoRARequest('verifier', 1, args.verifier)

    # 1단계: 베이스(무어댑터)로 n개 풀이 생성
    gen_prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in val['question']]
    outs = llm.generate(gen_prompts, SamplingParams(
        n=args.n, temperature=0.7, top_p=0.8, max_tokens=2048, seed=42))

    # 2단계: 검증자 어댑터로 각 풀이 채점 (첫 토큰 Yes 확률)
    flat_prompts, meta = [], []
    for i, o in enumerate(outs):
        for j, c in enumerate(o.outputs):
            flat_prompts.append(tok.apply_chat_template(
                [{'role': 'system', 'content': VERIFY_PROMPT},
                 {'role': 'user', 'content': 'Problem:' + chr(10) + val['question'][i] +
                  chr(10) * 2 + 'Proposed solution:' + chr(10) + c.text.strip()[:4000]}],
                tokenize=False, add_generation_prompt=True))
            meta.append((i, j))
    scores = llm.generate(flat_prompts, SamplingParams(
        temperature=0, max_tokens=1, logprobs=10), lora_request=verifier)

    score_map = defaultdict(dict)
    for (i, j), s in zip(meta, scores):
        lp = s.outputs[0].logprobs[0] if s.outputs[0].logprobs else None
        score_map[i][j] = p_yes(lp, tok)

    res = {'majority': 0, 'verifier_weighted': 0, 'best_of_1': 0}
    for i, o in enumerate(outs):
        answers = [extract_answer(c.text) for c in o.outputs]
        gold = int(val['answer'][i])
        votes = [a for a in answers if a is not None]
        maj = Counter(votes).most_common(1)[0][0] if votes else 0
        wsum = defaultdict(float)
        best_j, best_s = 0, -1.0
        for j, a in enumerate(answers):
            s = score_map[i].get(j, 0.0)
            if a is not None:
                wsum[a] += s
                if s > best_s:
                    best_s, best_j = s, j
        vw = max(wsum, key=wsum.get) if wsum else maj
        b1 = answers[best_j] if answers[best_j] is not None else maj
        res['majority'] += int(maj == gold)
        res['verifier_weighted'] += int(vw == gold)
        res['best_of_1'] += int(b1 == gold)

    n = len(val)
    out = {k: v / n for k, v in res.items()}
    for k, v in out.items():
        print(f'[{k}] {v:.1%} ({res[k]}/{n})')
    os.makedirs('results', exist_ok=True)
    json.dump(out, open('results/eval_bestofn.json', 'w'), indent=2)
    print('저장: results/eval_bestofn.json')


if __name__ == '__main__':
    main()
