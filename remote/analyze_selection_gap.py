"""선택 여지(selection gap) 분해 — pass@n과 현재 스택 사이의 12%p가 어디에 있는지.

exp37b(쌍대 비교)가 +1문항에 그친 뒤, "심판을 더 잘 만들면 얼마나 벌 수 있는가"의
상한을 재기 위해 만든 분석. GPU 불필요(덤프된 표본만 사용, 수 초).

핵심 질문 두 가지:
  ① 정답이 상위 k개 후보 안에 들어 있는 비율은? (재순위화로 도달 가능한 천장)
  ② 접전 문항에 한정하면 그 천장은 얼마인가? (토너먼트가 실제로 건드리는 범위)

사용: uv run python remote/analyze_selection_gap.py --samples results/val_samples.jsonl
"""
import argparse
import json
import math
from collections import Counter, defaultdict


def weighted_vote(samples, scale=2.0):
    """현재 최종 스택과 동일한 규칙 — 확신도(평균 토큰 logprob) 가중 다수결."""
    v = [(s['ans'], s['logp']) for s in samples if s['ans'] is not None]
    if not v:
        return None
    m = max(lp for _, lp in v)
    w = defaultdict(float)
    for a, lp in v:
        w[a] += math.exp((lp - m) * scale)
    return max(w, key=w.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='results/val_samples.jsonl')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--margin', type=int, default=3, help='접전 판정 기준 (exp37b와 동일)')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples, encoding='utf-8')]
    total = len(rows)
    c = Counter()
    cont = Counter()

    for r in rows:
        ss = r['samples'][:args.n]
        gold = r['gold']
        ranked = Counter(s['ans'] for s in ss if s['ans'] is not None).most_common()
        top = [a for a, _ in ranked]
        wv = weighted_vote(ss)

        c['weighted'] += (wv == gold)
        c['top1'] += (bool(top) and top[0] == gold)
        for k in (2, 3, 5):
            c[f'in_top{k}'] += (gold in top[:k])
        c['pass_n'] += (gold in top)

        # 접전 부분집합 — 토너먼트가 실제로 개입하는 범위
        if len(ranked) >= 2 and ranked[0][1] - ranked[1][1] <= args.margin:
            cont['n'] += 1
            cont['weighted'] += (wv == gold)
            cont['in_top2'] += (gold in top[:2])

    print(f'검증 {total}문항, n={args.n}')
    labels = [('weighted', '현재 스택(확신도 가중 투표)'), ('top1', '단순 다수결 1위'),
              ('in_top2', '정답이 상위 2후보 안'), ('in_top3', '정답이 상위 3후보 안'),
              ('in_top5', '정답이 상위 5후보 안'), ('pass_n', f'정답이 {args.n}개 중 어디든(pass@{args.n})')]
    for k, lab in labels:
        print('  %-32s %6.1f%%  (%d/%d)' % (lab, c[k] / total * 100, c[k], total))

    print(chr(10) + '완벽한 선택기를 가정했을 때의 상한:')
    for k, lab in [('in_top2', '상위 2후보 중 완벽 선택'),
                   ('in_top3', '상위 3후보 중 완벽 선택'),
                   ('pass_n', f'pass@{args.n} 전체(완전한 신탁)')]:
        d = c[k] - c['weighted']
        print('  %-28s %+d문항 (%+.2f%%p)' % (lab, d, d / total * 100))

    print(chr(10) + f'접전(1위-2위 득표차 <= {args.margin}) 부분집합:')
    print(f'  접전 문항                        {cont["n"]}개')
    print(f'  그중 현재 스택이 맞히는 수        {cont["weighted"]}개')
    print(f'  그중 정답이 상위 2후보 안         {cont["in_top2"]}개')
    d = cont['in_top2'] - cont['weighted']
    print('  → 완벽한 심판의 토너먼트 상한     %+d문항 (%+.2f%%p)' % (d, d / total * 100))
    print(chr(10) + '해석: 위 상한과 exp37b 실측(+1문항)의 차이가 곧 "심판 품질"의 여지다.')


if __name__ == '__main__':
    main()
