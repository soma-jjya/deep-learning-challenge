"""CCD(arXiv:2602.18232) 계열의 **전제**를 우리 모델에서 검증한다. GPU 불필요.

왜 방법이 아니라 전제부터 재는가: CCD 구현은 vLLM 밖에서 KV 캐시 두 개를 굴리는
커스텀 디코딩 루프라 며칠짜리 작업이다. 그런데 그 전제 — **"오류가 저확신 토큰에
국소화된다"** — 는 long-CoT thinking 모델의 긴 숙고 흐름에서 측정된 것이고,
우리 모델은 중앙값 388토큰짜리 short-CoT다. 전제가 성립하지 않으면 구현은 낭비다.
(exp55에서 '오답에 잘림이 12배 몰려 있다'는 상관을 인과로 읽고 GPU를 쓴 전례가 있다.)

전제가 참이라면 정답 풀이와 오답 풀이 사이에 아래 차이가 보여야 한다:
  ① 오답 풀이에 저확신 토큰이 더 많다
  ② 그 저확신 지점이 특정 구간에 몰린다(국소화) — 흩어져 있으면 개입 지점을 못 고른다
  ③ 저확신 정도로 정답/오답을 어느 정도 가를 수 있다(AUC)
③이 0.5 근처면 이 계열 전체가 우리에게 무의미하다 — 자기검증 5종이 전부 0.6에서
막힌 것과 같은 벽이기 때문이다.

사용: uv run python remote/test_confidence_premise.py --samples results/val_tok_s42.jsonl
"""
import argparse
import json
import math
import statistics
from collections import defaultdict


def weighted_vote(ss, scale=2.0):
    acc = defaultdict(float)
    for s in ss:
        a = s.get('ans')
        if a is not None:
            acc[a] += math.exp(scale * s.get('logp', 0.0))
    return max(acc, key=acc.get) if acc else None


def auc(pos, neg):
    """정답 쪽 점수가 오답 쪽보다 큰 쌍의 비율 (Mann-Whitney U)."""
    if not pos or not neg:
        return float('nan')
    pos = sorted(pos)
    wins = ties = 0
    import bisect
    for v in neg:
        lo = bisect.bisect_left(pos, v)
        hi = bisect.bisect_right(pos, v)
        wins += len(pos) - hi
        ties += hi - lo
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='results/val_tok_s42.jsonl')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--low-q', type=float, default=0.1,
                    help='이 분위 이하를 "저확신 토큰"으로 본다')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples, encoding='utf-8')]
    print(f'검증 {len(rows)}문항 × n={args.n}')

    # 전역 분위 기준을 먼저 잡는다 (풀이마다 기준이 다르면 비교가 안 된다)
    allv = []
    for r in rows:
        for s in r['samples'][:args.n]:
            allv.extend(s.get('tlp') or [])
    allv.sort()
    if not allv:
        raise SystemExit('토큰 logprob이 비어 있다 — 덤프를 확인할 것')
    thr = allv[int(len(allv) * args.low_q)]
    print(f'저확신 임계값(하위 {args.low_q:.0%} 분위): logprob {thr:.3f}')
    print(f'전체 토큰 {len(allv):,}개, 중앙값 {allv[len(allv)//2]:.3f}')

    # ── 풀이 단위: 정답 풀이 vs 오답 풀이 ──
    frac_ok, frac_ng = [], []      # ① 저확신 토큰 비율
    disp_ok, disp_ng = [], []      # ② 국소화 정도 (저확신 위치의 표준편차/길이)
    min_ok, min_ng = [], []        # ③ 최악 국소 구간 (하위 10% 평균)
    for r in rows:
        for s in r['samples'][:args.n]:
            lp = s.get('tlp') or []
            a = s.get('ans')
            if len(lp) < 20 or a is None:
                continue
            low_idx = [i for i, v in enumerate(lp) if v <= thr]
            frac = len(low_idx) / len(lp)
            k = max(1, len(lp) // 10)
            worst = sum(sorted(lp)[:k]) / k
            disp = (statistics.pstdev([i / len(lp) for i in low_idx])
                    if len(low_idx) > 1 else 0.0)
            if a == r['gold']:
                frac_ok.append(frac); disp_ok.append(disp); min_ok.append(worst)
            else:
                frac_ng.append(frac); disp_ng.append(disp); min_ng.append(worst)

    print()
    print(f'정답 풀이 {len(frac_ok):,}개 / 오답 풀이 {len(frac_ng):,}개')
    print(f'① 저확신 토큰 비율   정답 {statistics.mean(frac_ok):.3f} vs 오답 '
          f'{statistics.mean(frac_ng):.3f}   (오답이 크면 전제와 일치)')
    print(f'② 저확신 위치 분산   정답 {statistics.mean(disp_ok):.3f} vs 오답 '
          f'{statistics.mean(disp_ng):.3f}   (작을수록 국소화 — 0.29면 균등분포)')
    print(f'③ 최악 10% 평균      정답 {statistics.mean(min_ok):.3f} vs 오답 '
          f'{statistics.mean(min_ng):.3f}')

    print()
    print('판별력(AUC, 0.5=무신호):')
    for name, ok, ng in (('저확신 비율(적을수록 정답)', [-x for x in frac_ok], [-x for x in frac_ng]),
                         ('최악 10% 평균', min_ok, min_ng),
                         ('평균 logprob(현행 가중)', None, None)):
        if ok is None:
            mo = [s['logp'] for r in rows for s in r['samples'][:args.n]
                  if s.get('ans') == r['gold']]
            mn = [s['logp'] for r in rows for s in r['samples'][:args.n]
                  if s.get('ans') is not None and s['ans'] != r['gold']]
            print(f'  {name:26s} {auc(mo, mn):.3f}')
        else:
            print(f'  {name:26s} {auc(ok, ng):.3f}')

    print()
    print('해석:')
    print('  · ①에서 차이가 거의 없으면 "오류가 저확신에 국소화된다"는 전제가 우리 모델에')
    print('    성립하지 않는다 → CCD 계열 구현은 낭비다.')
    print('  · ②가 0.29(균등분포) 근처면 개입할 지점을 고를 수 없다.')
    print('  · AUC가 현행 평균 logprob과 비슷하면, 신호를 새로 얻은 것이 아니라')
    print('    이미 쓰고 있는 신호를 다른 이름으로 부른 것이다(자기검증 5종의 0.6 벽).')


if __name__ == '__main__':
    main()
