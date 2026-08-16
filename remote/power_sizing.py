"""검정력 기반 검증셋 크기 산정 — "몇 문항이어야 +0.6%p를 잡는가". GPU 불필요.

왜: 우리는 지금까지 판정 임계를 **관측된 불일치쌍에서 사후 계산**해왔는데(exp65),
그건 검정력이 아니다. 검정력은 "참 효과가 실재할 때 그것을 검출할 확률"이고,
483문항에서는 대응 비교를 써도 참 +2%p조차 검출 확률이 0.3 미만이다.

이 사실이 우리 실험사에 소급 적용된다: 부호가 일관 양수였으나 기각한 것들
(Budget Forcing +0.6%p 2회, 검증자 +0.4~0.5%p)은 **"효과 없음"이 아니라 "측정 불가"** 였다.

여기서는 몬테카를로로 (문항 수 × 참효과 × 불일치율) 격자의 검정력을 계산해,
**+0.6%p를 검정력 0.8로 잡으려면 몇 문항이 필요한가**에 답한다.

모형: 대응 비교. 두 방법이 d 비율의 문항에서만 다르고(불일치), 그중 참 효과 delta에
해당하는 만큼 한쪽으로 기운다. b ~ Binomial(n*d, 0.5 + delta/(2d)), c = n*d - b.
McNemar 정규근사로 유의 판정.

사용: python remote/power_sizing.py
"""
import argparse
import math
import random


def power(n, delta, disc, trials=4000, alpha=0.05, rng=None):
    """n문항, 참효과 delta(비율), 불일치율 disc에서 McNemar 검정력."""
    rng = rng or random.Random(0)
    nd = n * disc
    if nd < 1:
        return 0.0
    # 불일치쌍 중 A가 이기는 비율. delta = disc * (2p - 1)  →  p = 0.5 + delta/(2*disc)
    p = 0.5 + delta / (2 * disc)
    if not (0.0 < p < 1.0):
        return float('nan')
    z_crit = 1.959964
    hits = 0
    for _ in range(trials):
        m = int(round(nd))
        b = sum(1 for _ in range(m) if rng.random() < p)
        c = m - b
        if b + c == 0:
            continue
        z = (abs(b - c) - 1) / math.sqrt(b + c)
        if z > z_crit:
            hits += 1
    return hits / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=4000)
    args = ap.parse_args()
    rng = random.Random(20260816)

    # 우리 실측 불일치율: 32:0 vs 24:8에서 11/483 = 2.3%, vs 0:32에서 23/483 = 4.8%.
    # 개입이 클수록 불일치가 늘어난다. BF는 모든 표본을 이어쓰므로 그보다 클 것이다.
    DISCS = [0.05, 0.10, 0.20]
    DELTAS = [0.006, 0.010, 0.015, 0.020]   # +0.6 / +1.0 / +1.5 / +2.0 %p
    NS = [483, 1000, 2500, 5000, 8000, 12000]

    print('McNemar 검정력 (α=0.05 양측, 몬테카를로 {}회)'.format(args.trials))
    print()
    for disc in DISCS:
        print(f'── 불일치율 {disc:.0%} ' + '─' * 46)
        print(f'{"문항":>7s}' + ''.join(f'{d*100:+7.1f}%p' for d in DELTAS))
        for n in NS:
            row = f'{n:7d}'
            for d in DELTAS:
                pw = power(n, d, disc, args.trials, rng=rng)
                mark = '*' if pw >= 0.8 else (' ' if pw >= 0.5 else ' ')
                row += f'{pw:8.2f}{mark}'
            print(row)
        print()

    print('* = 검정력 0.8 이상')
    print()
    print('읽는 법:')
    print('  · 우리 성공 기준(+1.5%p)을 483문항에서 검출할 확률이 얼마인지 보라.')
    print('  · Budget Forcing의 관측 효과(+0.6%p)를 잡으려면 몇 문항이 필요한지 보라.')
    print('  · 불일치율이 낮을수록(두 방법이 비슷할수록) 같은 효과를 잡기 쉽다 —')
    print('    작은 개입일수록 오히려 검정력이 유리하다는 점이 직관에 반한다.')
    print()
    print('⚠️ 이 계산은 **문항 표집 불확실성만** 다룬다. 우리 실측 시드 간 변동')
    print('   (같은 설정 재실행 ±1%p)은 별도이며, 그것까지 넣으면 필요 문항 수는 더 커진다.')
    print('   대응 비교는 같은 시드를 쓰므로 시드 변동이 상쇄되지만, 채택 결정은')
    print('   여러 시드에서 부호가 일관되는지도 함께 봐야 한다.')


if __name__ == '__main__':
    main()
