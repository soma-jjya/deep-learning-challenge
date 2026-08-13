"""답 값 사전(answer prior)을 이용한 집계 — 39종 집계 규칙이 놓친 축. GPU 불필요.

지금까지 시험한 집계 규칙 39종은 전부 **확신도·잘림·길이** 기반이었다. 정작 후보 답의
**값 자체**는 한 번도 보지 않았다. 그런데 검증 표본을 재보면 신호가 뚜렷하다(n=32 기준):

           |값| 중앙  |값| 90%   |값|>10000 비율   답=0 비율
  정답        25        400          1.87%          0.72%
  오답        40      3,988          7.12%          2.39%

오답이 터무니없이 큰 값일 확률이 약 4배다(관측 최대값은 300자리 정수). 계산 실수가
자릿수 폭발로 이어지기 때문으로 보이며, 이는 **정답에는 없고 오답에만 있는 특징**이다.

과적합 방지: 검증 483문항은 작다. 경험적으로 곡선을 적합시키면 노이즈를 학습한다.
따라서 **모수 1~2개짜리 원리적 형태만** 시험하고, 판정은 시드 42/43/44 대응 비교로 한다
(exp48 교훈 — 단일 시드로 채택하면 오판한다).

사용: python remote/sweep_answer_prior.py --n 32
"""
import argparse
import json
import math
from collections import defaultdict


def softmax_w(logps, scale):
    m = max(logps)
    return [math.exp((lp - m) * scale) for lp in logps]


def plaus(a, mode, k):
    """답 값 a의 그럴듯함 가중치 (0~1). 클수록 그럴듯하다."""
    if mode == 'none':
        return 1.0
    m = abs(a)
    if mode == 'log':          # 자릿수에 비례해 완만히 감쇠
        return 1.0 / (1.0 + math.log10(1 + m) / k)
    if mode == 'digits':       # k자리를 넘어서면 급격히 감쇠
        d = len(str(m))
        return 1.0 if d <= k else 0.5 ** (d - k)
    if mode == 'hardcap':      # 10^k 초과는 강한 페널티(완전 배제는 아님)
        return 1.0 if m <= 10 ** k else 0.05
    raise ValueError(mode)


def vote(samples, scale=2.0, mode='none', k=3.0, zero_pen=1.0):
    v = [(s['ans'], s['logp']) for s in samples if s['ans'] is not None]
    if not v:
        return None
    w = defaultdict(float)
    for (a, _), ww in zip(v, softmax_w([lp for _, lp in v], scale)):
        p = plaus(a, mode, k)
        if a == 0:
            p *= zero_pen
        w[a] += ww * p
    return max(w, key=w.get) if w else None


def score(rows, n, **kw):
    return sum(1 for r in rows if vote(r['samples'][:n], **kw) == r['gold'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--files', nargs='+', default=[
        'results/val_samples.jsonl', 'results/val_samples_s43.jsonl', 'results/val_samples_s44.jsonl'])
    ap.add_argument('--labels', nargs='+', default=['seed42', 'seed43', 'seed44'])
    args = ap.parse_args()

    sets = [[json.loads(l) for l in open(f, encoding='utf-8')] for f in args.files]
    T = len(sets[0])

    variants = [('기준(가중 투표, 사전 없음)', dict(mode='none'))]
    for k in (1.0, 2.0, 3.0, 5.0):
        variants.append((f'log 감쇠 k={k}', dict(mode='log', k=k)))
    for k in (4, 5, 6, 7):
        variants.append((f'{k}자리 초과 감쇠', dict(mode='digits', k=k)))
    for k in (4, 5, 6):
        variants.append((f'10^{k} 초과 강페널티', dict(mode='hardcap', k=float(k))))
    variants.append(('0답 페널티만(0.5배)', dict(mode='none', zero_pen=0.5)))
    variants.append(('5자리 감쇠 + 0답 페널티', dict(mode='digits', k=5, zero_pen=0.5)))
    variants.append(('10^5 강페널티 + 0답 페널티', dict(mode='hardcap', k=5.0, zero_pen=0.5)))

    base = [score(s, args.n, mode='none') for s in sets]
    print(f'검증 {T}문항, n={args.n} — 시드 3개 대응 비교 (exp48 원칙)')
    print(f'{"변형":32s} ' + ' '.join(f'{l:>9s}' for l in args.labels) + f'{"평균차이":>12s} {"부호일치":>9s}')
    print('-' * 92)

    results = []
    for name, kw in variants:
        sc = [score(s, args.n, **kw) for s in sets]
        d = [a - b for a, b in zip(sc, base)]
        mean = sum(d) / len(d)
        allpos = all(x > 0 for x in d)
        results.append((mean, allpos, name, sc, d, kw))
        print(f'{name:32s} ' + ' '.join(f'{v:>9d}' for v in sc) +
              f'{mean:>+11.2f}문항 {"예" if allpos else "아니오":>9s}')

    print()
    winners = [r for r in results if r[1] and r[0] >= 3]
    if winners:
        best = max(winners, key=lambda r: r[0])
        print(f'✅ 채택 후보: {best[2]}  평균 {best[0]:+.2f}문항, 3시드 모두 양수')
        print(f'   설정: {best[5]}')
    else:
        print('❌ 3시드 모두 양수이면서 평균 +3문항 이상인 변형 없음 — 채택 기준 미달')


if __name__ == '__main__':
    main()
