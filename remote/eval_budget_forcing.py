"""Budget Forcing (s1 방식) — 모델이 답을 내려 할 때 "Wait"를 붙여 더 생각하게 만든다.

근거: s1: Simple test-time scaling (arXiv:2501.19393, EMNLP 2025).
  모델이 생성을 끝내려 할 때 EOS 대신 "Wait"를 이어붙이면 스스로 재검토·자기수정을 하고,
  이것만으로 경시 수학 정확도가 크게 올랐다(Qwen2.5-32B 기준). "Hmm" 등 중립어보다 "Wait"가 우수.

우리 상황에서의 의미: 지금까지 모든 시도는 '표를 세는 방식'이었고 개별 풀이의 질은 고정이었다.
이 기법은 **표 하나하나의 질을 올리는** 유일하게 남은 규칙 적합 축이다 (모델 출력만 사용, 도구 없음).

사용:
    uv run python remote/eval_budget_forcing.py --rounds 1 --n 1     # greedy + 1회 재검토
    uv run python remote/eval_budget_forcing.py --rounds 1 --n 8     # SC와 결합
"""
import argparse
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
    'Put your final integer answer inside ' + BS + 'boxed{}.')

WAIT = chr(10) + 'Wait, let me double-check my reasoning and the arithmetic.' + chr(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=1, help='"Wait" 재검토 횟수')
    ap.add_argument('--n', type=int, default=1, help='1이면 greedy, >1이면 SC 결합')
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--max-tokens', type=int, default=1200, help='라운드당 토큰 예산')
    # 확장 검증셋(2,500) 지원 — 483문항 대응비교 임계가 ±1.49%p인데 BF의 관측 효과가
    # +0.6%p라 지금 자로는 원리적으로 판별이 안 된다(exp65). 임계 ±0.65%p로 재판정한다.
    ap.add_argument('--val-ids', default=None, help='id,answer 컬럼 CSV (확장 검증셋)')
    ap.add_argument('--seed', type=int, default=42)
    # 문항별 표본을 저장해야 ① 확신도 가중 투표로 채점하고 ② McNemar 대응 비교를 할 수 있다.
    # 기존 스크립트는 요약 정확도만 남겨서 대응 비교가 불가능했다.
    ap.add_argument('--out', default=None, help='표본 덤프 경로 (val_samples 형식)')
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    if args.val_ids:
        want = pd.read_csv(args.val_ids)
        val = df[df['id'].isin(set(want['id']))].reset_index(drop=True)
        print(f'확장 검증셋: {args.val_ids}')
    else:
        val = df.sample(500, random_state=123).reset_index(drop=True)
        bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
        if os.path.exists(bad):
            bad_ids = set(pd.read_csv(bad)['id'])
            val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    print(f'검증 {len(val)}문항 / rounds={args.rounds} / n={args.n} / seed={args.seed}')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=8192)
    tok = llm.get_tokenizer()

    base = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in val['question']]

    greedy = args.n == 1
    # ⚠️ logprobs=0 필수 — 빠뜨리면 vLLM이 cumulative_logprob을 None으로 돌려주고
    # 확신도 가중 투표가 통째로 무너진다(exp37이 이 누락으로 0/65건을 호출해놓고
    # '가설 기각'으로 기록될 뻔했다).
    sp = SamplingParams(n=args.n, temperature=0 if greedy else args.temp,
                        top_p=1.0 if greedy else 0.8,
                        max_tokens=args.max_tokens, seed=args.seed, logprobs=0)

    # 1라운드: 평소대로 생성. 이후 라운드: 각 표본 뒤에 "Wait"를 붙여 이어서 생성.
    # 표본별로 텍스트를 누적해야 하므로 (문제 × 표본)을 평탄화해 관리한다.
    prompts = []
    for p in base:
        prompts.extend([p] * args.n if args.n > 1 else [p])
    # n>1이면 첫 라운드는 한 프롬프트에서 n개를 뽑는 게 효율적
    outs = llm.generate(base, sp)
    texts = [[c.text for c in o.outputs] for o in outs]        # [문제][표본]
    # 확신도는 라운드를 넘어 누적한다 — 최종 표본의 평균 토큰 logprob이 되도록
    # (총 logprob / 총 토큰). 라운드마다 갈아치우면 재검토 부분만 반영돼 왜곡된다.
    cum_lp = [[c.cumulative_logprob or 0.0 for c in o.outputs] for o in outs]
    cum_tk = [[max(1, len(c.token_ids)) for c in o.outputs] for o in outs]
    trunc = [[c.finish_reason == 'length' for c in o.outputs] for o in outs]

    acc_by_round = {}

    def score(round_idx):
        preds = []
        for ts in texts:
            answers = [extract_answer(t) for t in ts]
            v = [a for a in answers if a is not None]
            preds.append(Counter(v).most_common(1)[0][0] if v else 0)
        hit = sum(int(p) == int(a) for p, a in zip(preds, val['answer']))
        acc = hit / len(val)
        print(f'[round {round_idx}] 정확도 {acc:.1%} ({hit}/{len(val)})')
        acc_by_round[f'round{round_idx}'] = acc

    score(0)

    cont_sp = SamplingParams(n=1, temperature=0 if greedy else args.temp,
                             top_p=1.0 if greedy else 0.8,
                             max_tokens=args.max_tokens, seed=args.seed + 1, logprobs=0)
    for r in range(1, args.rounds + 1):
        flat_prompts, idx = [], []
        for i, ts in enumerate(texts):
            for j, t in enumerate(ts):
                flat_prompts.append(base[i] + t.rstrip() + WAIT)
                idx.append((i, j))
        conts = llm.generate(flat_prompts, cont_sp)
        for (i, j), o in zip(idx, conts):
            # 이어붙인 재검토까지 포함한 전체 텍스트로 갱신 (마지막 boxed가 최종 답이 됨)
            c = o.outputs[0]
            texts[i][j] = texts[i][j].rstrip() + WAIT + c.text
            cum_lp[i][j] += (c.cumulative_logprob or 0.0)
            cum_tk[i][j] += max(1, len(c.token_ids))
            trunc[i][j] = c.finish_reason == 'length'
        score(r)

    os.makedirs('results', exist_ok=True)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            for i in range(len(val)):
                samples = [{'ans': extract_answer(texts[i][j]),
                            'logp': cum_lp[i][j] / cum_tk[i][j],
                            'trunc': trunc[i][j], 'len': cum_tk[i][j]}
                           for j in range(len(texts[i]))]
                f.write(json.dumps({'id': val['id'][i], 'gold': int(val['answer'][i]),
                                    'samples': samples}, ensure_ascii=False) + chr(10))
        print(f'표본 덤프 저장: {args.out} — analyze_selection_gap.py로 가중 채점 가능')

    tag = f'bf_r{args.rounds}_n{args.n}'
    json.dump(acc_by_round, open(f'results/eval_{tag}.json', 'w'), indent=2)
    print('저장:', f'results/eval_{tag}.json')


if __name__ == '__main__':
    main()
