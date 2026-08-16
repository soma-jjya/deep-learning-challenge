"""표차 보상(Certified Self-Consistency, arXiv 2510.17472)의 전제 검증. GPU 불필요.

## 왜 구현 전에 전제부터 재는가
이 기법은 답 히스토그램만으로 계산되는 **라벨 없는** RL 보상이다:
  SNR   r = (N1−N2)² / [n(N1+N2) − (N1−N2)²]      ← 1·2위 표차 최대화
  엔트로피 r = Σ (Nj/n) log(Nj/n)                    ← 답 분포 엔트로피 최소화
자기검증 6종이 부딪힌 판별력 0.6 벽이 **적용되지 않는다**(보상에 correctness 항이 없다).
그래서 구조적으로 새롭다. 구현 비용은 GRPO 6~8시간이다.

그런데 기제를 따져보면 의심스럽다:
  **분포를 현재 1위 쪽으로 날카롭게 만들면 argmax는 바뀌지 않는다.**
  이미 A가 1위인 문항에서 A를 더 뾰족하게 해도 여전히 A가 뽑힌다 → 정확도 불변.
  오히려 정답이 2위인 문항(우리 실패의 15%)에서는 **오답 1위를 더 굳힌다.**

여기서는 우리 실제 덤프로 그 직관을 정량 확인한다:
  ① 분포를 인위적으로 날카롭게(온도 하강) 만들면 정확도가 어떻게 변하는가
  ② 반대로 평평하게 만들면?
  ③ 1위가 정답인 문항 / 오답인 문항에서 각각 어떻게 갈리는가

①이 평평하면 이 계열 전체가 우리에게 무의미하다 — 표차는 늘지만 argmax가 안 바뀌므로
얻는 것은 **실행 간 분산 축소**뿐이고, 그건 정확도가 아니다.

사용: uv run python remote/test_sharpening_premise.py
"""
import argparse
import json
import math
from collections import defaultdict


def vote(ss, scale, sharpen):
    """확신도 가중 투표. sharpen>1이면 표 분포를 날카롭게(승자 쏠림) 만든다."""
    c = defaultdict(float)
    for s in ss:
        a = s.get('ans')
        if a is not None:
            c[a] += math.exp(scale * s.get('logp', 0.0))
    if not c:
        return None, 0.0, 0.0
    if sharpen != 1.0:
        c = {k: v ** sharpen for k, v in c.items()}
    tot = sum(c.values()) or 1.0
    ranked = sorted(c.values(), reverse=True)
    n1 = ranked[0] / tot
    n2 = (ranked[1] / tot) if len(ranked) > 1 else 0.0
    return max(c, key=c.get), n1, n1 - n2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='+',
                    default=['results/val_samples.jsonl',
                             'results/val_samples_s43.jsonl',
                             'results/val_samples_s44.jsonl'])
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()

    SHARPENS = [0.25, 0.5, 1.0, 2.0, 4.0, 16.0]
    print(f'표 분포를 날카롭게/평평하게 만들었을 때의 정확도 (n={args.n})')
    print()
    print(f'{"sharpen":>9s}{"의미":>14s}' + ''.join(f'{"s"+s.split("_s")[-1][:2] if "_s" in s else "s42":>8s}'
                                                    for s in args.samples)
          + f'{"평균":>9s}{"평균표차":>10s}')
    print('-' * 76)

    base_mean = None
    for sh in SHARPENS:
        accs, margins = [], []
        for path in args.samples:
            rows = [json.loads(l) for l in open(path, encoding='utf-8')]
            ok = 0
            mg = []
            for r in rows:
                pred, _, margin = vote(r['samples'][:args.n], 2.0, sh)
                ok += (pred == r['gold'])
                mg.append(margin)
            accs.append(ok)
            margins.append(sum(mg) / len(mg))
        m = sum(accs) / len(accs)
        if sh == 1.0:
            base_mean = m
        label = '평평하게' if sh < 1 else ('현행' if sh == 1 else '날카롭게')
        print(f'{sh:9.2f}{label:>14s}' + ''.join(f'{a:8d}' for a in accs)
              + f'{m:9.1f}{sum(margins)/len(margins):10.3f}')

    print()
    print(f'※ 기준(sharpen=1.0) = {base_mean:.1f}문항')
    print()

    # 1위 정답/오답으로 쪼개서 확인
    print('1위가 정답인 문항 vs 오답인 문항에서 sharpen의 효과 (seed 42)')
    rows = [json.loads(l) for l in open(args.samples[0], encoding='utf-8')]
    right = [r for r in rows if vote(r['samples'][:args.n], 2.0, 1.0)[0] == r['gold']]
    wrong = [r for r in rows if vote(r['samples'][:args.n], 2.0, 1.0)[0] != r['gold']]
    print(f'  1위 정답 {len(right)}문항 / 1위 오답 {len(wrong)}문항')
    for sh in (0.5, 1.0, 4.0, 16.0):
        a = sum(1 for r in right if vote(r['samples'][:args.n], 2.0, sh)[0] == r['gold'])
        b = sum(1 for r in wrong if vote(r['samples'][:args.n], 2.0, sh)[0] == r['gold'])
        print(f'  sharpen {sh:5.2f}: 정답 유지 {a}/{len(right)}, 오답에서 회복 {b}/{len(wrong)}'
              f'  → 합계 {a+b}')

    print()
    print('해석:')
    print('  · sharpen을 키워도 정확도가 평평하면, 표차 보상이 얻는 것은 정확도가 아니라')
    print('    **실행 간 분산 축소**뿐이다. argmax가 안 바뀌기 때문이다.')
    print('  · 1위 오답 문항에서 회복이 0이면, 날카롭게 만드는 것은 우리 실패 구간을')
    print('    건드리지 못한다 — 오히려 오답 1위를 더 굳힌다.')


if __name__ == '__main__':
    main()
