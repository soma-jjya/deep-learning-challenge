"""exp74 — reasoning-clustered SC의 **전제**를 텍스트 없이 대리 진단. GPU 0.

## 가설 (제안된 것)
SC는 32표를 독립 투표처럼 세지만, 실제로는 **같은 잘못된 사고방식을 조금씩 다르게 말한**
샘플이 여러 개일 수 있다. 그렇다면 답별 '독립 추론 클러스터 수'로 세면 결과가 뒤집힐 수 있다.

    wrong A : 12표지만 실은 2개 클러스터
    gold    :  7표지만 실은 4개 클러스터   → gold 채택

## 전제가 참이라면 텍스트 없이도 흔적이 보여야 한다
같은 오류 모드에서 나온 샘플들은 **서로 닮아야** 한다. 우리 덤프에는 풀이 텍스트가 없지만
**길이와 평균 logprob**은 있다. 오답 표들이 정답 표들보다 **균질**하다면(분산이 작다면)
"동일 오류 모드" 가설과 정합하고, 텍스트를 재생성해 클러스터링할 값어치가 생긴다.

반대로 둘의 균질성이 비슷하거나 오답 쪽이 더 흩어져 있다면, 텍스트를 뽑아도(GPU 45분)
얻을 것이 없다.

## 측정 (문항 내 대응 비교 — 문항별 난이도 효과를 상쇄)
회수 가능 구간(SC 오답이고 정답이 표본에 있는 문항)에서, 같은 문항 안의
  · 지배적 오답을 낸 표들의 길이 표준편차 / logprob 표준편차
  · 정답을 낸 표들의 같은 값
을 짝지어 비교한다.

사용: python remote/diag_error_mode_homogeneity.py
"""
import argparse
import json
import math
from collections import defaultdict


def wvote(ss, scale=2.0):
    c = defaultdict(float)
    for s in ss:
        a = s.get('ans')
        if a is not None:
            c[a] += math.exp(scale * s.get('logp', 0.0))
    return max(c, key=c.get) if c else None


def sd(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def cv(xs):
    """변동계수 — 길이처럼 척도가 다른 값을 비교하기 위해."""
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    s = sd(xs)
    return s / m if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='results/val_samples.jsonl')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--min-votes', type=int, default=3,
                    help='양쪽 다 이 표 수 이상인 문항만 (분산 추정 안정)')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples, encoding='utf-8')]
    pairs = []
    for r in rows:
        ss = r['samples'][:args.n]
        gold = r['gold']
        pred = wvote(ss)
        if pred == gold or pred is None:
            continue                       # SC가 맞은 문항은 대상 아님
        g = [s for s in ss if s.get('ans') == gold]
        w = [s for s in ss if s.get('ans') == pred]
        if len(g) < args.min_votes or len(w) < args.min_votes:
            continue
        pairs.append({
            'n_gold': len(g), 'n_wrong': len(w),
            'len_cv_gold': cv([s['len'] for s in g]),
            'len_cv_wrong': cv([s['len'] for s in w]),
            'lp_sd_gold': sd([s['logp'] for s in g]),
            'lp_sd_wrong': sd([s['logp'] for s in w]),
        })

    print(f'회수 가능 구간에서 양쪽 {args.min_votes}표 이상인 문항: {len(pairs)}개')
    if not pairs:
        print('표본 부족 — min-votes를 낮춰 볼 것')
        return
    print(f'  (평균 정답 {sum(p["n_gold"] for p in pairs)/len(pairs):.1f}표 / '
          f'지배오답 {sum(p["n_wrong"] for p in pairs)/len(pairs):.1f}표)')
    print()

    print('가설: 오답 표들이 "동일 오류 모드"라면 정답 표들보다 **균질**해야 한다')
    print(f'{"지표":22s}{"정답 표":>10s}{"지배오답 표":>12s}{"오답이 더 균질":>14s}')
    print('-' * 60)
    for label, kg, kw in (('풀이 길이 변동계수', 'len_cv_gold', 'len_cv_wrong'),
                          ('logprob 표준편차', 'lp_sd_gold', 'lp_sd_wrong')):
        vg = [p[kg] for p in pairs if p[kg] is not None and p[kw] is not None]
        vw = [p[kw] for p in pairs if p[kg] is not None and p[kw] is not None]
        if not vg:
            continue
        mg, mw = sum(vg) / len(vg), sum(vw) / len(vw)
        wins = sum(1 for a, b in zip(vg, vw) if b < a)     # 오답 쪽이 더 작으면(균질) 승
        print(f'{label:22s}{mg:10.4f}{mw:12.4f}{wins:>8d}/{len(vg):<5d}'
              f' = {wins/len(vg):.1%}')

    print()
    print('해석:')
    print('  · "오답이 더 균질"이 60% 이상이고 평균 차이도 뚜렷하면 동일 오류 모드 가설과')
    print('    정합한다 → 텍스트를 재생성해 클러스터링할 값어치가 있다.')
    print('  · 50% 부근이면 오답 표들이 정답 표들보다 특별히 닮지 않았다는 뜻이고,')
    print('    텍스트 클러스터링으로 얻을 것이 없을 가능성이 크다.')
    print()
    print('⚠️ 길이·logprob은 추론 방식의 대리 지표일 뿐이다. 이 진단이 음성이어도')
    print('   "텍스트 클러스터가 전혀 없다"를 증명하지는 않는다 — 다만 GPU를 쓰기 전')
    print('   기대값을 낮출 근거는 된다.')


if __name__ == '__main__':
    main()
