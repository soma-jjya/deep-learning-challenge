"""① 소수 지분 혼합 투표 스윕 ② 우리 판정 절차의 실제 판별력(McNemar). GPU 불필요.

## ① 소수 지분 혼합
exp56은 혼합을 **50:50과 32+32 합집합**으로만 했다. 그런데 "분포가 다른 생성기를 섞어
오류를 탈상관시킨다"는 논증은 **소수 지분에서만** 성립하는 예측이다 — 소수라야 베이스가
커버하는 문제의 다수결을 흔들지 않고, 베이스가 못 푸는 문제에서만 캐스팅보트를 쥔다.
50:50은 그 조건을 위반한다(어댑터 단독이 −8.67문항이므로 절반을 주면 그 손실을 절반 떠안는다).
비율 스윕: 32:0(대조) / 28:4 / 24:8 / 20:12 / 16:16.

## ② 우리 판정 절차의 실제 판별력
"483문항에서 SE ≈ ±1.9%p이므로 +1~2%p는 판별 불가"라는 지적은 **독립 두 측정**을 비교할 때의
값이다. 우리는 같은 문항·같은 시드로 **대응 비교**를 해왔으므로 관련 분산은 주변부 SE가 아니라
**불일치쌍(discordant pairs)** 에서 나온다. McNemar로 실제 임계를 계산해, 우리가 "노이즈"로
기각한 실험들이 정말 검정력 부족이었는지 확인한다.

사용: uv run python remote/analyze_mix_and_power.py
"""
import argparse
import json
import math
import os
from collections import defaultdict


def wvote(ss, scale=2.0):
    c = defaultdict(float)
    for s in ss:
        a = s.get('ans')
        if a is not None:
            c[a] += math.exp(scale * s.get('logp', 0.0))
    return max(c, key=c.get) if c else None


def load(p):
    return {json.loads(l)['id']: json.loads(l) for l in open(p, encoding='utf-8')}


def correct_vec(A, B, ids, kb, kt):
    """베이스 kb표 + 어댑터 kt표로 투표했을 때 문항별 정오 벡터."""
    out = []
    for i in ids:
        ss = A[i]['samples'][:kb] + (B[i]['samples'][:kt] if kt else [])
        out.append(1 if wvote(ss) == A[i]['gold'] else 0)
    return out


def mcnemar(x, y):
    """x, y는 같은 문항에 대한 정오 벡터(1/0). (b, c, diff, p근사, 최소검출차이)"""
    b = sum(1 for u, v in zip(x, y) if u == 0 and v == 1)   # y만 맞힘
    c = sum(1 for u, v in zip(x, y) if u == 1 and v == 0)   # x만 맞힘
    n = b + c
    diff = b - c
    if n == 0:
        return b, c, diff, float('nan'), float('nan')
    # 정규근사 (연속성 보정)
    z = (abs(diff) - 1) / math.sqrt(n) if n > 0 else 0.0
    p = math.erfc(max(0.0, z) / math.sqrt(2))
    # 유의(α=0.05, z=1.96)해지려면 필요한 |diff|
    mdd = 1.96 * math.sqrt(n) + 1
    return b, c, diff, p, mdd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', default=['42', '43', '44'])
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()

    def base_path(s):
        return 'results/val_samples.jsonl' if s == '42' else f'results/val_samples_s{s}.jsonl'

    print('=' * 74)
    print('① 소수 지분 혼합 투표 (총 32표, 베이스:어댑터 비율)')
    print('=' * 74)
    ratios = [(32, 0), (28, 4), (24, 8), (20, 12), (16, 16), (0, 32)]
    per_ratio = defaultdict(list)
    vecs = {}
    for s in args.seeds:
        bp, tp = base_path(s), f'results/val_samples_teacher_full_s{s}.jsonl'
        if not (os.path.exists(bp) and os.path.exists(tp)):
            print(f'  seed {s}: 덤프 없음 — 건너뜀')
            continue
        A, B = load(bp), load(tp)
        ids = sorted(set(A) & set(B))
        for kb, kt in ratios:
            v = correct_vec(A, B, ids, kb, kt)
            per_ratio[(kb, kt)].append(sum(v))
            vecs[(s, kb, kt)] = v
        print(f'  seed {s}: 공통 {len(ids)}문항')

    print()
    hdr = f'{"비율(base:adapter)":22s}' + ''.join(f'seed {s:>4s}' for s in args.seeds) + f'{"평균":>10s}{"대조대비":>10s}'
    print(hdr)
    print('-' * len(hdr))
    base_mean = None
    for kb, kt in ratios:
        vals = per_ratio.get((kb, kt))
        if not vals:
            continue
        m = sum(vals) / len(vals)
        if base_mean is None:
            base_mean = m
        print(f'{f"{kb}:{kt}":22s}' + ''.join(f'{v:9d}' for v in vals) +
              f'{m:10.1f}{m-base_mean:+10.2f}')

    # 대응 비교로 유의성까지
    print()
    print('대조(32:0) 대비 McNemar (시드별):')
    for kb, kt in ratios[1:]:
        line = f'  {kb}:{kt:<4d} '
        for s in args.seeds:
            if (s, kb, kt) not in vecs:
                continue
            b, c, d, p, mdd = mcnemar(vecs[(s, 32, 0)], vecs[(s, kb, kt)])
            line += f'  seed{s}: b={b:3d} c={c:3d} diff={d:+3d} p={p:.3f}'
        print(line)

    print()
    print('=' * 74)
    print('② 우리 판정 절차의 실제 판별력 — "483문항이면 ±1.9%p"가 맞는가')
    print('=' * 74)
    # 주변부 SE (독립 두 측정을 비교할 때)
    p0 = 0.758
    n0 = 483
    se_marginal = math.sqrt(p0 * (1 - p0) / n0)
    print(f'  독립 측정 1개의 SE      : {se_marginal:.4f} = ±{se_marginal*100:.2f}%p')
    print(f'  독립 두 측정의 차이 SE  : {se_marginal*math.sqrt(2)*100:.2f}%p'
          f'  ← 대응 비교를 안 쓰면 이 값이 임계')
    print()
    print('  그러나 우리는 같은 문항·같은 시드로 대응 비교를 해왔다.')
    print('  대응 비교의 임계는 불일치쌍 수에서 나온다:')
    # 실제 우리 비교들의 불일치쌍
    for kb, kt in [(16, 16), (0, 32)]:
        for s in args.seeds[:1]:
            if (s, kb, kt) not in vecs:
                continue
            b, c, d, p, mdd = mcnemar(vecs[(s, 32, 0)], vecs[(s, kb, kt)])
            print(f'    예: 32:0 vs {kb}:{kt} (seed {s}) — 불일치쌍 {b+c}개, '
                  f'유의해지려면 |diff| ≥ {mdd:.1f}문항 (= {mdd/483*100:.2f}%p)')
    print()
    print('  → 대응 비교 임계가 독립 비교보다 낮으면, "노이즈로 기각"이 검정력 부족이')
    print('    아니었다는 뜻이다. 반대면 지적이 맞고 검증셋 확장이 필요하다.')


if __name__ == '__main__':
    main()
