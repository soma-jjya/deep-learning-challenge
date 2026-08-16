"""분야별로 최적 집계 규칙이 다른가 — 어려운 문제에 다른 접근이 필요하다는 가설의 검증.

배경: 분야별 진단에서 정확도가 극단적으로 갈렸다 — 기타/문장제 85.5%(전체의 69%) vs
조합론 30.0%. 그런데 우리가 시도한 집계 규칙 63종은 전부 **모든 문항에 같은 규칙**을 썼다.
exp50이 조건부 규칙을 시도했지만 분기 기준이 **표차**였지 **분야**가 아니었다.

가설: 합의가 잘 되는 문장제와, 표가 흩어지는 경시형은 최적 규칙이 다를 수 있다.
쉬운 쪽은 이미 잘 맞히므로 건드릴 이유가 없고, 어려운 쪽만 다른 규칙을 쓰면 이득이 날 여지.

⚠️ 함정 주의: 분야별로 최적 규칙을 고르면 **자유도가 분야 수만큼 늘어난다.** 9개 분야에서
각각 7개 규칙 중 최고를 고르면 우연히 좋아 보이는 조합이 반드시 나온다. 그래서
① 분야별 표본 수를 함께 출력하고 ② 시드 3개에서 **같은 규칙이 이기는지** 확인한다.
한 시드에서만 이기는 조합은 채택하지 않는다.

사용: uv run python remote/sweep_category_rules.py
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def r_majority(ss):
    c = defaultdict(float)
    for s in ss:
        if s.get('ans') is not None:
            c[s['ans']] += 1
    return max(c, key=c.get) if c else None


def r_weighted(ss, scale=2.0):
    c = defaultdict(float)
    for s in ss:
        if s.get('ans') is not None:
            c[s['ans']] += math.exp(scale * s.get('logp', 0.0))
    return max(c, key=c.get) if c else None


def r_weighted4(ss):
    return r_weighted(ss, 4.0)


def r_weighted1(ss):
    return r_weighted(ss, 1.0)


def r_drop_trunc(ss):
    keep = [s for s in ss if not s.get('trunc')]
    return r_weighted(keep or ss)


def r_trim25(ss):
    ok = [s for s in ss if s.get('ans') is not None]
    if not ok:
        return None
    ok.sort(key=lambda s: s.get('logp', 0.0), reverse=True)
    keep = ok[:max(1, int(len(ok) * 0.75))]
    return r_weighted(keep)


def r_short(ss):
    """짧은 풀이 우선 — 오답 풀이가 길다는 관찰(843 vs 411)의 집계판."""
    ok = [s for s in ss if s.get('ans') is not None]
    if not ok:
        return None
    ok.sort(key=lambda s: s.get('len', 0))
    return r_weighted(ok[:max(1, len(ok) // 2)])


RULES = {'majority': r_majority, 'weighted': r_weighted, 'weighted_s1': r_weighted1,
         'weighted_s4': r_weighted4, 'drop_trunc': r_drop_trunc,
         'trim25': r_trim25, 'short_half': r_short}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='+',
                    default=['results/val_samples.jsonl',
                             'results/val_samples_s43.jsonl',
                             'results/val_samples_s44.jsonl'])
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()

    from analyze_by_category import classify
    import pandas as pd

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    qmap = {r.id: str(r.question) for r in df.itertuples()}

    # 시드별 · 분야별 · 규칙별 정답 수
    per_seed = []
    for path in args.samples:
        rows = [json.loads(l) for l in open(path, encoding='utf-8')]
        tab = defaultdict(lambda: defaultdict(int))
        cnt = defaultdict(int)
        for r in rows:
            cat = classify(qmap.get(r['id'], ''))
            cnt[cat] += 1
            ss = r['samples'][:args.n]
            for name, fn in RULES.items():
                if fn(ss) == r['gold']:
                    tab[cat][name] += 1
        per_seed.append((os.path.basename(path), tab, cnt))

    cats = sorted(per_seed[0][2], key=lambda c: -per_seed[0][2][c])
    print(f'집계 규칙 {len(RULES)}종 × 분야 {len(cats)}개 × 시드 {len(per_seed)}개')
    print()
    hdr = f'{"분야":12s}{"n":>5s}  ' + '  '.join(f'{k[:11]:>11s}' for k in RULES)
    print(hdr)
    print('-' * len(hdr))

    winners = defaultdict(list)
    for cat in cats:
        n = per_seed[0][2][cat]
        # 시드 평균으로 표시
        vals = {}
        for name in RULES:
            vals[name] = sum(t[cat][name] for _, t, _ in per_seed) / len(per_seed)
        best = max(vals, key=vals.get)
        row = f'{cat:12s}{n:5d}  ' + '  '.join(f'{vals[k]:11.1f}' for k in RULES)
        print(row + f'   ← {best}')
        # 시드별 승자도 기록 (한 시드만의 우연인지 보기 위해)
        for sname, t, _ in per_seed:
            w = max(RULES, key=lambda k: t[cat][k])
            winners[cat].append(w)

    print()
    print('시드별 승자 일관성 (3시드 모두 같은 규칙이 이겨야 신뢰):')
    stable = []
    for cat in cats:
        ws = winners[cat]
        mark = '✅' if len(set(ws)) == 1 else '  '
        if len(set(ws)) == 1 and ws[0] != 'weighted':
            stable.append((cat, ws[0]))
        print(f'  {mark} {cat:12s} {" / ".join(ws)}')

    print()
    if stable:
        print('3시드 일관되게 weighted를 이긴 분야:')
        for cat, w in stable:
            print(f'  {cat} → {w}')
        print('⚠️ 그래도 자유도가 분야 수만큼 늘어난 선택이다. 이득의 크기가 시드 노이즈')
        print('   (문항 기준 ±5)보다 확실히 커야 하고, 최종 판정은 리더보드다.')
    else:
        print('3시드 일관되게 weighted를 이긴 분야 없음 → 분야별 규칙 분기는 근거 없음.')


if __name__ == '__main__':
    main()
