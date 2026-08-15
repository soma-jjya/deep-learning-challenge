"""토큰 단위 확신도를 투표 가중치로 써본다 — GPU 불필요.

전제 검증(test_confidence_premise.py)에서 '최악 10% 평균'의 판별력이 AUC 0.610으로
현행 평균 logprob(0.591)보다 **약간 높게** 나왔다. 차이는 작지만 공짜로 확인할 수 있으므로
실제 투표 정확도로 옮겨지는지 본다.

⚠️ 기대치는 낮게 잡는다. exp51에서 시뮬레이션이 판별력 0.65에 +1.45%p를 예측했는데 실제는
−0.83%p였다 — 보상 오차가 투표 오차와 **상관되어 있으면** 판별력이 이득으로 바뀌지 않는다.
여기서도 두 신호가 같은 logprob에서 나오므로 상관이 매우 높을 것이다.

사용: uv run python remote/sweep_token_conf_vote.py --samples results/val_tok_s42.jsonl
"""
import argparse
import json
import math
from collections import defaultdict


def vote(rows, keyfn, n, scale):
    ok = 0
    for r in rows:
        acc = defaultdict(float)
        for s in r['samples'][:n]:
            a = s.get('ans')
            if a is None:
                continue
            acc[a] += math.exp(scale * keyfn(s))
        if acc and max(acc, key=acc.get) == r['gold']:
            ok += 1
    return ok


def worst_frac(s, q):
    lp = s.get('tlp') or []
    if not lp:
        return s.get('logp', 0.0)
    k = max(1, int(len(lp) * q))
    return sum(sorted(lp)[:k]) / k


def tail_mean(s, m):
    lp = s.get('tlp') or []
    if not lp:
        return s.get('logp', 0.0)
    t = lp[-m:] if len(lp) > m else lp
    return sum(t) / len(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='results/val_tok_s42.jsonl')
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples, encoding='utf-8')]
    N = len(rows)
    print(f'검증 {N}문항 × n={args.n}')
    print()

    base = vote(rows, lambda s: s.get('logp', 0.0), args.n, 2.0)
    print(f'  {"현행: 평균 logprob (scale 2.0)":38s} {base/N:6.1%}  ({base}/{N})   기준')

    variants = []
    for q in (0.05, 0.10, 0.20):
        variants.append((f'최악 {int(q*100)}% 평균', lambda s, q=q: worst_frac(s, q)))
    for m in (32, 128):
        variants.append((f'마지막 {m}토큰 평균', lambda s, m=m: tail_mean(s, m)))
    variants.append(('최저 토큰 1개', lambda s: min(s.get('tlp') or [s.get('logp', 0.0)])))
    variants.append(('평균 + 최악10% 절반씩',
                     lambda s: 0.5 * s.get('logp', 0.0) + 0.5 * worst_frac(s, 0.10)))

    for name, fn in variants:
        best = None
        for scale in (0.5, 1.0, 2.0, 4.0):
            ok = vote(rows, fn, args.n, scale)
            if best is None or ok > best[0]:
                best = (ok, scale)
        ok, scale = best
        print(f'  {name+f" (최적 scale {scale})":38s} {ok/N:6.1%}  ({ok}/{N})   {ok-base:+d}문항')

    print()
    print('⚠️ exp48 원칙: 시드 1개로 채택하지 말 것. 그리고 여러 scale 중 최고를 고른')
    print('   값이므로 이 표는 이미 낙관적이다 — 채택 후보가 나오면 scale을 고정해')
    print('   시드 43·44로 재확인해야 한다.')


if __name__ == '__main__':
    main()
