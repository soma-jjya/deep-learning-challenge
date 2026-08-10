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


# ── exp44: trim_lowconf 정밀 탐색 변형들 ──────────────────────────
def trunc_then_trim(ss, frac=0.25, base=majority):
    """② 잘린 풀이(trunc) 먼저 제외한 뒤, 남은 표에서 확신도 하위 frac을 추가로 버림."""
    kept = [s for s in ss if not s['trunc']]
    if not kept:
        kept = ss
    return trim_lowconf(kept, frac, base)


def length_trim(ss, low_pct=10, high_pct=90, base=majority):
    """④ 문항 내에서 비정상적으로 짧거나(low_pct 미만) 긴(high_pct 초과) 풀이를 제외."""
    v = [s for s in ss if s['ans'] is not None]
    if len(v) < 4:
        return base(ss)
    lens = sorted(s['len'] for s in v)

    def pct(p):
        idx = min(int(len(lens) * p / 100), len(lens) - 1)
        return lens[idx]

    lo, hi = pct(low_pct), pct(high_pct)
    kept = [s for s in v if lo <= s['len'] <= hi]
    return base(kept if len(kept) >= 2 else v)


def trim_with_fallback(ss, frac=0.25, min_votes=3, base=majority):
    """⑤ trim 후 최다 득표가 min_votes 미만(합의 실패)이면 원본 전체 표로 폴백."""
    v = [s for s in ss if s['ans'] is not None]
    if len(v) < 4:
        return base(ss)
    v_sorted = sorted(v, key=lambda s: s['logp'], reverse=True)
    keep = v_sorted[:max(2, int(len(v_sorted) * (1 - frac)))]
    top_votes = Counter(s['ans'] for s in keep).most_common(1)[0][1]
    return base(v) if top_votes < min_votes else base(keep)


def build_strategies():
    trim_fracs = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]
    strategies = {
        'majority (기준)': majority,
        'weighted s=2.0': lambda ss: weighted(ss, 2.0),
        'drop_truncated': drop_truncated,
        'prefer_short(동률시)': prefer_short,
        'tiebreak_by_conf': tiebreak_by_conf,
        'plausible_only': plausible_only,
    }
    # ① trim 비율 스윕
    for f in trim_fracs:
        strategies[f'trim_lowconf {int(f * 100)}%'] = (lambda f: lambda ss: trim_lowconf(ss, f))(f)
    # ② trim 후 가중 결합
    for f in trim_fracs:
        strategies[f'trim{int(f * 100)}%+weighted'] = (
            lambda f: lambda ss: trim_lowconf(ss, f, lambda x: weighted(x, 2.0)))(f)
    # ③ 잘린 풀이 제외 + trim 조합
    for f in [0.10, 0.25, 0.40]:
        strategies[f'trunc제외+trim{int(f * 100)}%'] = (lambda f: lambda ss: trunc_then_trim(ss, f))(f)
    # ④ 길이 기반 trim (다른 퍼센타일 폭)
    for lo, hi in [(10, 90), (5, 95), (20, 80)]:
        strategies[f'length_trim {lo}-{hi}pct'] = (lambda lo, hi: lambda ss: length_trim(ss, lo, hi))(lo, hi)
    # ⑤ trim 후 최소 득표 미달 시 전체 표로 폴백
    for mv in [2, 3, 4]:
        strategies[f'trim25%+fallback(min{mv})'] = (
            lambda mv: lambda ss: trim_with_fallback(ss, 0.25, mv))(mv)
    return strategies


def run(rows, strategies):
    total = len(rows)
    results = {}
    for name, fn in strategies.items():
        hit = sum(1 for r in rows if fn(r['samples']) == r['gold'])
        results[name] = hit
    return results, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default='results/val_samples.jsonl')
    ap.add_argument('--n', type=int, default=None, help='앞에서 n개만 사용 (표본 수 효과 확인)')
    ap.add_argument('--sweep-n', default=None,
                     help='콤마구분 n 목록 — 지정 시 각 n에서 전체 전략 비교를 순차 실행 (예: 16,32,64)')
    args = ap.parse_args()

    all_rows = load(args.file)
    strategies = build_strategies()

    ns = [int(x) for x in args.sweep_n.split(',')] if args.sweep_n else [args.n]

    for n in ns:
        rows = [dict(r) for r in all_rows]
        if n:
            for r in rows:
                r['samples'] = r['samples'][:n]
        results, total = run(rows, strategies)
        nsamp = len(rows[0]['samples'])
        base_hit = results['majority (기준)']

        print(f'{total}문항 × {nsamp}표본으로 집계 전략 비교' + chr(10))
        print(f'{"전략":<28}{"정확도":>10}{"맞힌수":>9}{"기준대비":>10}')
        for name, hit in sorted(results.items(), key=lambda x: -x[1]):
            d = hit - base_hit
            flag = '  <-- 성공기준(+8)' if d >= 8 else ''
            print(f'{name:<28}{hit/total:>9.1%}{hit:>9}{d:>+10}{flag}')

        # 상한 참고: 표본 중 정답이 하나라도 있으면 맞힌 것으로 계산
        oracle = sum(1 for r in rows if any(s['ans'] == r['gold'] for s in r['samples']))
        print(chr(10) + f'참고 상한(pass@{nsamp}): {oracle/total:.1%} ({oracle}/{total}) '
              f'— 집계로 회수 가능한 최대치')
        print(chr(10) + ('=' * 70) + chr(10))


if __name__ == '__main__':
    main()
