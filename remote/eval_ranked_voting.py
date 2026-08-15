"""H28 — Ranked Voting Self-Consistency (arXiv 2505.10772, ACL 2025 Findings).

우리가 기각한 집계 규칙 63종은 전부 **투표지는 그대로 두고 개표 방식만** 바꾼 것이었다
(확신도 가중, trim_lowconf, drop_truncated, 길이 기반, 부트스트랩, 규칙 앙상블 …).
이 기법은 **투표지 자체**를 바꾼다 — 각 CoT가 답 하나가 아니라 **순위 매긴 후보 리스트**를
내게 하고, 그것을 Instant-Runoff / Borda / MRR로 집계한다.

왜 우리 병목에 맞는가: 다수결 1위가 틀린 119문항에서 정답의 위치는 2위 18.5%, 3위 12.6% —
합쳐서 **31%**다. 단일답 투표지에서는 이 표들이 아예 기록되지 않는다. 순위 투표지는
"여러 경로에서 꾸준히 2위"인 후보를 이기게 할 수 있다.

논문 근거: 6개 벤치마크에서 SC가 전부 최하위. **평가 체크포인트가 Qwen2.5-3B-Instruct로
우리와 동일**하다. 6개 평균 SC 60.13 → MRR 65.08 / IRV 64.76 / Borda 64.72.
자유형식 출력인 Date Understanding에서 47.97→60.70(+12.73pp)이 나와, 선택지 집합이 없어도
작동한다는 직접 증거가 된다.

⚠️ 논문은 퓨샷으로 형식을 가르치지만 우리는 퓨샷이 두 번 다 해로웠다(exp20 −1.6%p,
   exp57 −11문항). 그래서 **제로샷 지시**로 형식을 요구하고, 대신 **순응률을 반드시 측정**한다.
   순응률이 낮으면 이 실험은 무효다 — exp37이 logprobs 누락으로 0/65건을 호출해놓고
   "가설 기각"으로 기록될 뻔한 사고를 되풀이하지 않기 위한 장치다.

사용: uv run python remote/eval_ranked_voting.py --n 32 --seed 42
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import pandas as pd

BS = chr(92)
NL = chr(10)

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.' + NL * 2 +
    'After the boxed answer, output one last line ranking the answers you consider most '
    'likely, from most to least likely, in exactly this format:' + NL +
    'RANKING: <first> > <second> > <third>' + NL +
    'The first entry must equal your boxed answer. Give three distinct integers; if you are '
    'certain, still give the two next most plausible values a solver might arrive at.')

RANK_PAT = re.compile(r'RANKING\s*[:：]\s*(.+)')
INT_PAT = re.compile(r'-?[0-9][0-9,]*')


def parse_ranking(text, max_k=3):
    """마지막 RANKING 줄에서 정수 순위를 뽑는다. 실패하면 빈 리스트."""
    hits = RANK_PAT.findall(text)
    if not hits:
        return []
    out = []
    for tok in hits[-1].split('>'):
        m = INT_PAT.search(tok)
        if not m:
            continue
        try:
            v = int(m.group(0).replace(',', ''))
        except ValueError:
            continue
        if v not in out:
            out.append(v)
        if len(out) >= max_k:
            break
    return out


# ── 집계 규칙 ──────────────────────────────────────────────────────────────
def r_top1(ballots, _w):
    """대조군: 순위 1위만 세는 단순 다수결 (기존 스택과 같은 정보량)."""
    c = defaultdict(float)
    for b in ballots:
        if b:
            c[b[0]] += 1
    return max(c, key=c.get) if c else None


def r_top1_weighted(ballots, w):
    """대조군2: 1위만 세되 확신도 가중 — 현재 최종 스택과 동일."""
    c = defaultdict(float)
    for b, wt in zip(ballots, w):
        if b:
            c[b[0]] += wt
    return max(c, key=c.get) if c else None


def r_borda(ballots, _w):
    """k위에 (K-k+1)점. 2위 표가 처음으로 기록된다."""
    c = defaultdict(float)
    for b in ballots:
        K = len(b)
        for i, a in enumerate(b):
            c[a] += K - i
    return max(c, key=c.get) if c else None


def r_borda_weighted(ballots, w):
    c = defaultdict(float)
    for b, wt in zip(ballots, w):
        K = len(b)
        for i, a in enumerate(b):
            c[a] += (K - i) * wt
    return max(c, key=c.get) if c else None


def r_mrr(ballots, _w):
    """k위에 1/k점. 논문에서 6개 평균 최고였던 규칙."""
    c = defaultdict(float)
    for b in ballots:
        for i, a in enumerate(b):
            c[a] += 1.0 / (i + 1)
    return max(c, key=c.get) if c else None


def r_mrr_weighted(ballots, w):
    c = defaultdict(float)
    for b, wt in zip(ballots, w):
        for i, a in enumerate(b):
            c[a] += wt / (i + 1)
    return max(c, key=c.get) if c else None


def r_irv(ballots, _w):
    """즉석 결선: 1위표 최소 후보를 탈락시키고 그 표를 다음 순위로 넘긴다."""
    live = {a for b in ballots for a in b}
    if not live:
        return None
    while True:
        c = defaultdict(float)
        for b in ballots:
            for a in b:
                if a in live:
                    c[a] += 1
                    break
        if not c:
            return None
        total = sum(c.values())
        best = max(c, key=c.get)
        if c[best] * 2 > total or len(live) <= 1:
            return best
        loser = min(live, key=lambda a: c.get(a, 0.0))
        live.discard(loser)
        if not live:
            return best


RULES = {
    'top1 (단순 다수결)': r_top1,
    'top1_weighted (현행 스택)': r_top1_weighted,
    'borda': r_borda,
    'borda_weighted': r_borda_weighted,
    'mrr': r_mrr,
    'mrr_weighted': r_mrr_weighted,
    'irv': r_irv,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--scale', type=float, default=2.0, help='확신도 softmax 스케일')
    ap.add_argument('--out', default=None)
    ap.add_argument('--score-only', default=None, help='기존 덤프를 채점만 한다')
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from answer_extract import extract_answer

    out = args.out or f'results/val_samples_rank_s{args.seed}.jsonl'

    if args.score_only:
        out = args.score_only
    else:
        df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
        df.columns = df.columns.str.strip()
        val = df.sample(500, random_state=123).reset_index(drop=True)
        bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
        if os.path.exists(bad):
            val = val[~val['id'].isin(set(pd.read_csv(bad)['id']))].reset_index(drop=True)
        print(f'검증 {len(val)}문항 × {args.n}샘플, 순위 투표지 (seed={args.seed})')

        from vllm import LLM, SamplingParams
        llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
                  gpu_memory_utilization=0.85, max_model_len=4096)
        tok = llm.get_tokenizer()
        prompts = [tok.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': q}],
            tokenize=False, add_generation_prompt=True) for q in val['question']]
        outs = llm.generate(prompts, SamplingParams(
            n=args.n, temperature=args.temp, top_p=args.top_p,
            max_tokens=args.max_tokens, seed=args.seed, logprobs=0))

        os.makedirs('results', exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            for i, o in enumerate(outs):
                samples = []
                for c in o.outputs:
                    nt = max(1, len(c.token_ids))
                    boxed = extract_answer(c.text)
                    rank = parse_ranking(c.text)
                    samples.append({'ans': boxed, 'rank': rank,
                                    'logp': c.cumulative_logprob / nt,
                                    'trunc': c.finish_reason == 'length', 'len': nt})
                f.write(json.dumps({'id': val['id'][i], 'gold': int(val['answer'][i]),
                                    'samples': samples}, ensure_ascii=False) + NL)
        print(f'저장: {out}')

    rows = [json.loads(l) for l in open(out, encoding='utf-8')]

    # ── 순응률 먼저 ── 이 값이 낮으면 아래 정확도는 의미가 없다
    tot = sum(len(r['samples']) for r in rows)
    with_rank = sum(1 for r in rows for s in r['samples'] if s.get('rank'))
    ge2 = sum(1 for r in rows for s in r['samples'] if len(s.get('rank') or []) >= 2)
    mism = sum(1 for r in rows for s in r['samples']
               if s.get('rank') and s.get('ans') is not None and s['rank'][0] != s['ans'])
    print()
    print(f'순응률: RANKING 줄 있음 {with_rank}/{tot} = {with_rank/tot:.1%}')
    print(f'        2개 이상 순위   {ge2}/{tot} = {ge2/tot:.1%}   ← 이 값이 핵심')
    print(f'        1위 ≠ boxed     {mism}/{max(1,with_rank)} = {mism/max(1,with_rank):.1%}')
    if ge2 / tot < 0.5:
        print('⚠️ 2개 이상 순위를 낸 표본이 절반 미만이다 — 순위 투표가 실질적으로')
        print('   작동하지 않으므로 아래 수치로 가설을 기각하지 말 것 (형식 지시부터 재검토).')

    # ── 채점 ──
    print()
    base_ok = None
    for name, fn in RULES.items():
        ok = 0
        for r in rows:
            ss = r['samples'][:args.n]
            ballots, w = [], []
            for s in ss:
                b = s.get('rank') or ([s['ans']] if s.get('ans') is not None else [])
                if not b:
                    continue
                ballots.append(b)
                w.append(math.exp(args.scale * s.get('logp', 0.0)))
            if fn(ballots, w) == r['gold']:
                ok += 1
        if name.startswith('top1_weighted'):
            base_ok = ok
        print(f'  {name:26s} {ok/len(rows):6.1%}  ({ok}/{len(rows)})')
    if base_ok is not None:
        print()
        print(f'  ※ 기준은 같은 실행의 top1_weighted({base_ok}문항)다 — 다른 실행의 366과')
        print('    비교하면 vLLM 재실행 변동(exp34b 실측 0.48%p)이 섞인다.')
    print()
    print('⚠️ exp48 원칙: 시드 1개로 채택하지 말 것.')


if __name__ == '__main__':
    main()
