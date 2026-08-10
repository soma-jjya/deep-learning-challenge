"""집계 전략 스윕 — dump_samples.py가 저장한 표본으로 GPU 없이 수십 가지 규칙을 비교.

핵심 질문: "같은 표본에서, 답을 고르는 규칙만 바꿔 몇 문제를 더 건질 수 있는가?"
GPU 재생성이 필요 없으므로 수 초 만에 끝난다 — 지금까지 못 해본 규모의 탐색.

사용: uv run python remote/sweep_aggregation.py [--n 32]
"""
import argparse
import json
import math
from collections import Counter, defaultdict


def load(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]


def softmax_weights(logps, scale):
    m = max(logps)
    return [math.exp((lp - m) * scale) for lp in logps]


# ── 집계 전략들: (표본 리스트) → 최종 답 ─────────────────────────
def majority(ss):
    v = [s['ans'] for s in ss if s['ans'] is not None]
    return Counter(v).most_common(1)[0][0] if v else 0


def weighted(ss, scale=2.0):
    v = [(s['ans'], s['logp']) for s in ss if s['ans'] is not None]
    if not v:
        return 0
    w = defaultdict(float)
    for (a, lp), ww in zip(v, softmax_weights([x[1] for x in v], scale)):
        w[a] += ww
    return max(w, key=w.get)


def drop_truncated(ss, base=majority):
    kept = [s for s in ss if not s['trunc']]
    return base(kept if kept else ss)


def trim_lowconf(ss, frac=0.25, base=majority):
    """확신도 하위 frac을 버리고 집계 (품질 낮은 표를 솎아내기)."""
    v = [s for s in ss if s['ans'] is not None]
    if len(v) < 4:
        return base(ss)
    v.sort(key=lambda s: s['logp'], reverse=True)
    keep = v[:max(2, int(len(v) * (1 - frac)))]
    return base(keep)


def prefer_short(ss, base=majority):
    """같은 답끼리 묶었을 때 평균 길이가 짧은 답을 선호 (간결한 풀이가 정답일 확률)."""
    v = [s for s in ss if s['ans'] is not None]
    if not v:
        return 0
    cnt, length = Counter(), defaultdict(list)
    for s in v:
        cnt[s['ans']] += 1
        length[s['ans']].append(s['len'])
    top = cnt.most_common()
    best = top[0][1]
    tied = [a for a, c in top if c == best]
    if len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda a: sum(length[a]) / len(length[a]))


def tiebreak_by_conf(ss):
    """다수결 동률일 때 최고 확신도 표본의 답으로 결정."""
    v = [s for s in ss if s['ans'] is not None]
    if not v:
        return 0
    cnt = Counter(s['ans'] for s in v)
    top = cnt.most_common()
    best = top[0][1]
    tied = [a for a, c in top if c == best]
    if len(tied) == 1:
        return tied[0]
    bestconf = defaultdict(lambda: -1e9)
    for s in v:
        bestconf[s['ans']] = max(bestconf[s['ans']], s['logp'])
    return max(tied, key=lambda a: bestconf[a])


def plausible_only(ss, base=majority):
    """비상식적으로 큰 답(|x| > 10^12)을 무효표 처리 — 학습 데이터 답 범위 기반."""
    kept = [s for s in ss if s['ans'] is None or abs(s['ans']) <= 10**12]
    return base(kept if any(s['ans'] is not None for s in kept) else ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default='results/val_samples.jsonl')
    ap.add_argument('--n', type=int, default=None, help='앞에서 n개만 사용 (표본 수 효과 확인)')
    args = ap.parse_args()

    rows = load(args.file)
    if args.n:
        for r in rows:
            r['samples'] = r['samples'][:args.n]
    total = len(rows)
    nsamp = len(rows[0]['samples'])
    print(f'{total}문항 × {nsamp}표본으로 집계 전략 비교' + chr(10))

    strategies = {
        'majority (기준)': majority,
        'weighted s=1.0': lambda ss: weighted(ss, 1.0),
        'weighted s=2.0': lambda ss: weighted(ss, 2.0),
        'weighted s=4.0': lambda ss: weighted(ss, 4.0),
        'drop_truncated': drop_truncated,
        'trim_lowconf 25%': lambda ss: trim_lowconf(ss, 0.25),
        'trim_lowconf 50%': lambda ss: trim_lowconf(ss, 0.5),
        'trim+weighted': lambda ss: trim_lowconf(ss, 0.25, lambda x: weighted(x, 2.0)),
        'prefer_short(동률시)': prefer_short,
        'tiebreak_by_conf': tiebreak_by_conf,
        'plausible_only': plausible_only,
        'plausible+weighted': lambda ss: plausible_only(ss, lambda x: weighted(x, 2.0)),
    }

    results = {}
    for name, fn in strategies.items():
        hit = sum(1 for r in rows if fn(r['samples']) == r['gold'])
        results[name] = hit
    base_hit = results['majority (기준)']

    print(f'{"전략":<24}{"정확도":>10}{"맞힌수":>9}{"기준대비":>10}')
    for name, hit in sorted(results.items(), key=lambda x: -x[1]):
        d = hit - base_hit
        print(f'{name:<24}{hit/total:>9.1%}{hit:>9}{d:>+10}')

    # 상한 참고: 표본 중 정답이 하나라도 있으면 맞힌 것으로 계산
    oracle = sum(1 for r in rows if any(s['ans'] == r['gold'] for s in r['samples']))
    print(chr(10) + f'참고 상한(pass@{nsamp}): {oracle/total:.1%} ({oracle}/{total}) '
          f'— 집계로 회수 가능한 최대치')


if __name__ == '__main__':
    main()
