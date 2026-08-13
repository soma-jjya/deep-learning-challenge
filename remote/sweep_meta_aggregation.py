"""메타 집계 — 규칙을 바꾸는 대신 **규칙을 조합·조건부로 쓰는** 축. GPU 불필요.

exp44까지의 39종은 전부 "하나의 규칙을 전 문항에 똑같이 적용"했고 76.4%에서 포화했다.
여기서는 규칙 자체가 아니라 **규칙을 쓰는 방식**을 바꾼다 — 세 가지 미검증 축:

  ① 규칙 앙상블   : 서로 다른 규칙들이 각각 한 표씩 던져 최빈값을 택한다.
                    개별 규칙이 동등하게 포화했더라도 **틀리는 문항이 서로 다르면** 이득이 난다.
  ② 부트스트랩 안정성 : 32표를 복원추출로 B번 재표집해 매번 투표하고, 가장 자주 이긴 답을 택한다.
                    확신도 높은 단일 이상치가 표를 끌고 가는 것을 막는다(가중 투표의 약점).
  ③ 조건부 규칙   : 표차가 크면 다수결, 접전이면 다른 규칙 — 문항 난이도에 따라 규칙을 바꾼다.
                    exp44는 전 문항 동일 규칙만 봤으므로 이 축은 비어 있다.

판정은 exp48 원칙대로 **시드 42/43/44 대응 비교**. 단일 시드 개선은 채택하지 않는다.

사용: PYTHONIOENCODING=utf-8 python remote/sweep_meta_aggregation.py --n 32
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict


# ── 기본 규칙들 ────────────────────────────────────────────────
def _valid(ss):
    return [s for s in ss if s['ans'] is not None]


def r_majority(ss):
    v = _valid(ss)
    if not v:
        return None
    return Counter(s['ans'] for s in v).most_common(1)[0][0]


def r_weighted(ss, scale=2.0):
    v = _valid(ss)
    if not v:
        return None
    m = max(s['logp'] for s in v)
    w = defaultdict(float)
    for s in v:
        w[s['ans']] += math.exp((s['logp'] - m) * scale)
    return max(w, key=w.get)


def r_trim(ss, ratio=0.25):
    """확신도 하위 ratio를 버리고 다수결 (exp31b에서 로컬 최고 76.4%)."""
    v = sorted(_valid(ss), key=lambda s: s['logp'], reverse=True)
    if not v:
        return None
    keep = v[:max(1, int(len(v) * (1 - ratio)))]
    return Counter(s['ans'] for s in keep).most_common(1)[0][0]


def r_drop_trunc(ss):
    v = [s for s in _valid(ss) if not s['trunc']] or _valid(ss)
    if not v:
        return None
    return Counter(s['ans'] for s in v).most_common(1)[0][0]


def r_len_trim(ss):
    """비정상적으로 짧은 풀이(하위 20% 길이)를 제외."""
    v = _valid(ss)
    if not v:
        return None
    cut = sorted(s['len'] for s in v)[max(0, int(len(v) * 0.2)) - 1] if len(v) >= 5 else 0
    keep = [s for s in v if s['len'] >= cut] or v
    return Counter(s['ans'] for s in keep).most_common(1)[0][0]


MEMBERS = [('majority', r_majority),
           ('weighted2', lambda ss: r_weighted(ss, 2.0)),
           ('weighted1', lambda ss: r_weighted(ss, 1.0)),
           ('weighted4', lambda ss: r_weighted(ss, 4.0)),
           ('trim25', r_trim),
           ('drop_trunc', r_drop_trunc),
           ('len_trim', r_len_trim)]


# ── ① 규칙 앙상블 ──────────────────────────────────────────────
def m_rule_ensemble(ss, members=None):
    members = members or MEMBERS
    votes = [f(ss) for _, f in members]
    votes = [v for v in votes if v is not None]
    if not votes:
        return None
    c = Counter(votes).most_common()
    top = c[0][1]
    tied = [a for a, n in c if n == top]
    if len(tied) == 1:
        return tied[0]
    w = r_weighted(ss)              # 동점이면 기준 규칙이 결정
    return w if w in tied else tied[0]


# ── ② 부트스트랩 안정성 ────────────────────────────────────────
def m_bootstrap(ss, B=200, seed=0, scale=2.0):
    v = _valid(ss)
    if not v:
        return None
    rng = random.Random(seed)
    wins = Counter()
    for _ in range(B):
        samp = [v[rng.randrange(len(v))] for _ in range(len(v))]
        m = max(s['logp'] for s in samp)
        w = defaultdict(float)
        for s in samp:
            w[s['ans']] += math.exp((s['logp'] - m) * scale)
        wins[max(w, key=w.get)] += 1
    return wins.most_common(1)[0][0]


# ── ③ 조건부 규칙 ──────────────────────────────────────────────
def margin(ss):
    v = _valid(ss)
    c = Counter(s['ans'] for s in v).most_common(2)
    if len(c) < 2:
        return 99
    return c[0][1] - c[1][1]


def m_conditional(ss, thresh=3, wide=r_majority, close=None):
    close = close or (lambda s: r_weighted(s, 2.0))
    return wide(ss) if margin(ss) > thresh else close(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--files', nargs='+', default=[
        'results/val_samples.jsonl', 'results/val_samples_s43.jsonl', 'results/val_samples_s44.jsonl'])
    ap.add_argument('--labels', nargs='+', default=['seed42', 'seed43', 'seed44'])
    args = ap.parse_args()

    sets = [[json.loads(l) for l in open(f, encoding='utf-8')] for f in args.files]
    T = len(sets[0])
    N = args.n

    def sc(rows, fn):
        return sum(1 for r in rows if fn(r['samples'][:N]) == r['gold'])

    variants = [
        ('기준(가중 투표 scale=2)', lambda ss: r_weighted(ss, 2.0)),
        ('규칙 앙상블 (7종 투표)', m_rule_ensemble),
        ('규칙 앙상블 (다수결계 4종)', lambda ss: m_rule_ensemble(
            ss, [m for m in MEMBERS if m[0] in ('majority', 'trim25', 'drop_trunc', 'len_trim')])),
        ('부트스트랩 B=200', lambda ss: m_bootstrap(ss, 200, 0)),
        ('부트스트랩 B=500', lambda ss: m_bootstrap(ss, 500, 0)),
        ('조건부: 표차>3 다수결 / 접전 가중', lambda ss: m_conditional(ss, 3)),
        ('조건부: 표차>3 다수결 / 접전 trim25', lambda ss: m_conditional(ss, 3, r_majority, r_trim)),
        ('조건부: 표차>5 다수결 / 접전 부트스트랩',
         lambda ss: m_conditional(ss, 5, r_majority, lambda s: m_bootstrap(s, 200, 0))),
        ('조건부: 표차>1 trim25 / 접전 가중',
         lambda ss: m_conditional(ss, 1, r_trim, lambda s: r_weighted(s, 2.0))),
    ]

    base = [sc(s, lambda ss: r_weighted(ss, 2.0)) for s in sets]
    print(f'검증 {T}문항, n={N} — 시드 3개 대응 비교 (exp48 원칙)')
    print(f'{"변형":38s} ' + ' '.join(f'{l:>8s}' for l in args.labels) + f'{"평균차이":>12s} {"부호일치":>9s}')
    print('-' * 96)

    results = []
    for name, fn in variants:
        s = [sc(x, fn) for x in sets]
        d = [a - b for a, b in zip(s, base)]
        mean = sum(d) / len(d)
        allpos = all(x > 0 for x in d)
        results.append((mean, allpos, name, s, d))
        print(f'{name:38s} ' + ' '.join(f'{v:>8d}' for v in s) +
              f'{mean:>+11.2f}문항 {"예" if allpos else "아니오":>9s}')

    print()
    ok = [r for r in results if r[1] and r[0] >= 3]
    if ok:
        b = max(ok, key=lambda r: r[0])
        print(f'✅ 채택 후보: {b[2]}  평균 {b[0]:+.2f}문항, 3시드 모두 양수 (시드별 {b[4]})')
    else:
        near = max((r for r in results if r[2] != variants[0][0]), key=lambda r: r[0])
        print(f'❌ 채택 기준(3시드 양수 + 평균 +3문항) 충족 변형 없음.')
        print(f'   최고 근접: {near[2]} 평균 {near[0]:+.2f}문항, 시드별 {near[4]}')


if __name__ == '__main__':
    main()
