"""재순위화 시뮬레이션 — "쌍 정확도가 얼마여야 목표에 닿는가". GPU 불필요, 수 초.

보상 헤드(exp51)의 쌍 정확도가 0.62 근방에서 평평해졌다. 그런데 쌍 정확도와 실제
Best-of-N 정확도의 관계는 자명하지 않다 — 쌍 판별은 "둘 중 하나"지만 실제 과제는
"32개 중 하나"이고, 게다가 32개 중 정답이 하나도 없는 문항이 12%나 된다.

그래서 GPU를 더 쓰기 전에 상한을 먼저 계산한다. 기존 덤프에는 각 표본의 정오가
이미 들어 있으므로(ans vs gold), **쌍 정확도 a를 갖는 가상의 재순위화기**를 넣어
결과를 시뮬레이션할 수 있다.

모델: 후보 i의 점수 = z·[정답] + N(0,1).
      두 후보(정답 하나, 오답 하나)의 점수차는 N(z, 2)이므로
      P(정답이 더 높음) = Φ(z/√2) = a  →  z = √2·Φ⁻¹(a)

이 시뮬레이션이 답하는 것: 목표(+1.5%p)에 닿으려면 쌍 정확도가 얼마여야 하는가.
그 값이 도달 불가능하게 크면, 재순위화 평가에 GPU를 더 쓸 이유가 없다.

사용: PYTHONIOENCODING=utf-8 python remote/simulate_reranker.py
"""
import argparse
import json
import math
import random
import statistics
from collections import defaultdict


def phi_inv(p):
    """표준정규 분위수 (Acklam 근사) — scipy 없이 쓰기 위해."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def weighted_vote(cands, scale=2.0):
    v = [(c['ans'], c['logp']) for c in cands if c['ans'] is not None]
    if not v:
        return None
    m = max(lp for _, lp in v)
    w = defaultdict(float)
    for a, lp in v:
        w[a] += math.exp((lp - m) * scale)
    return max(w, key=w.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', default=[
        'results/val_samples.jsonl', 'results/val_samples_s43.jsonl', 'results/val_samples_s44.jsonl'])
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--trials', type=int, default=40, help='난수 반복 횟수 (평균을 취한다)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    sets = [[json.loads(l) for l in open(f, encoding='utf-8')] for f in args.files]
    T = len(sets[0])

    base = [sum(1 for r in s if weighted_vote(r['samples'][:args.n]) == r['gold']) for s in sets]
    base_mean = statistics.mean(base)
    print(f'검증 {T}문항, n={args.n}, 시드 {len(sets)}개 평균')
    print(f'기준(확신도 가중 투표): {base_mean:.1f}문항 ({base_mean/T:.1%})')
    print()
    print(f'{"쌍 정확도":>10s} {"Best-of-1":>14s} {"보상 가중투표":>16s} {"기준 대비(투표)":>18s}')
    print('-' * 62)

    rng = random.Random(args.seed)
    rows = []
    for a in (0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99):
        z = math.sqrt(2) * phi_inv(a)
        best_tot = vote_tot = 0
        for s in sets:
            for _ in range(args.trials):
                b = v = 0
                for r in s:
                    cands = [c for c in r['samples'][:args.n] if c['ans'] is not None]
                    if not cands:
                        continue
                    sc = [z * (c['ans'] == r['gold']) + rng.gauss(0, 1) for c in cands]
                    # Best-of-1: 최고 점수 후보의 답
                    b += cands[max(range(len(cands)), key=lambda i: sc[i])]['ans'] == r['gold']
                    # 보상 가중 투표
                    m = max(sc)
                    w = defaultdict(float)
                    for c, x in zip(cands, sc):
                        w[c['ans']] += math.exp(x - m)
                    v += max(w, key=w.get) == r['gold']
                best_tot += b
                vote_tot += v
        denom = len(sets) * args.trials
        bm, vm = best_tot / denom, vote_tot / denom
        d = vm - base_mean
        rows.append((a, bm, vm, d))
        print(f'{a:>10.2f} {bm/T:>13.1%} {vm/T:>15.1%} {d:>+14.1f}문항 ({d/T*100:+.2f}%p)')

    print()
    need = next((r for r in rows if r[3] / T * 100 >= 1.5), None)
    if need:
        print(f'→ +1.5%p에 닿으려면 쌍 정확도가 최소 **{need[0]:.2f}** 필요.')
    else:
        print('→ 쌍 정확도 0.99에서도 +1.5%p에 닿지 않는다.')
    print(f'   현재 exp51 관측치는 0.62 근방 — 위 표에서 해당 행을 보라.')


if __name__ == '__main__':
    main()
