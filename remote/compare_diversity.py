"""표본 다양성 비교 — "학습은 왜 SC를 망가뜨리는가"를 추론이 아니라 측정으로 확인.

exp09b에서 SFT가 greedy(−1.9%p)보다 SC(−2.9%p)를 더 크게 떨어뜨린 것을 보고
"학습이 풀이 다양성을 죽인다"고 진단했지만, 그건 정확도 차이에서 역추론한 것이었다.
exp47(DPO)도 같은 패턴(greedy +1.2%p / SC −0.6%p)을 보였으므로, 이번엔 두 덤프를
직접 비교해 기전을 확정한다.

핵심 지표
  pass@n      : 정답이 n개 표본 중 한 번이라도 나오는 비율. **떨어지면 능력 자체의 손실**
                (다양성이 줄어 정답에 도달할 경로를 잃음)
  고유답 개수  : 문제당 서로 다른 답의 수. 줄면 분포가 좁아진 것
  1위 득표율   : 최다득표 답의 비율. 오르면 분포가 뾰족해진 것

pass@n은 유지되는데 1위 득표율만 오르면 → 단순한 "분포 첨예화"(선택엔 오히려 유리할 수 있음)
pass@n이 함께 떨어지면 → "능력 손실". SC 스택에 치명적.

사용: uv run python remote/compare_diversity.py --base results/val_samples.jsonl \
        --other results/val_samples_dpo.jsonl --label-other DPO
"""
import argparse
import json
from collections import Counter


def stats(path, n):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    total = len(rows)
    acc = Counter()
    distinct = share = 0.0
    for r in rows:
        ss = r['samples'][:n]
        gold = r['gold']
        ans = [s['ans'] for s in ss if s['ans'] is not None]
        cnt = Counter(ans)
        acc['pass_n'] += (gold in cnt)
        acc['top1'] += (bool(cnt) and cnt.most_common(1)[0][0] == gold)
        acc['none'] += (not ans)          # 답 추출 자체 실패
        distinct += len(cnt)
        if ans:
            share += cnt.most_common(1)[0][1] / len(ans)
    return {
        'n_problems': total,
        'pass_n': acc['pass_n'] / total,
        'top1_acc': acc['top1'] / total,
        'distinct_avg': distinct / total,
        'top1_share': share / total,
        'no_answer': acc['none'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='results/val_samples.jsonl')
    ap.add_argument('--other', required=True)
    ap.add_argument('--label-other', default='어댑터')
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()

    b = stats(args.base, args.n)
    o = stats(args.other, args.n)

    rows = [
        (f'pass@{args.n} (정답이 한 번이라도 등장)', 'pass_n', True),
        ('단순 다수결 1위 정확도', 'top1_acc', True),
        ('문제당 고유 답 개수 (평균)', 'distinct_avg', False),
        ('1위 답 득표 비율 (평균)', 'top1_share', True),
    ]
    print(f'표본 비교 (n={args.n}) — 베이스 vs {args.label_other}')
    print(f'{"지표":38s} {"베이스":>10s} {args.label_other:>10s} {"차이":>10s}')
    for lab, key, pct in rows:
        d = o[key] - b[key]
        if pct:
            print(f'{lab:38s} {b[key]*100:9.1f}% {o[key]*100:9.1f}% {d*100:+9.2f}%p')
        else:
            print(f'{lab:38s} {b[key]:10.2f} {o[key]:10.2f} {d:+10.2f}')
    print(f'{"답 추출 실패 문항 수":38s} {b["no_answer"]:10d} {o["no_answer"]:10d} '
          f'{o["no_answer"]-b["no_answer"]:+10d}')

    dp = (o['pass_n'] - b['pass_n']) * 100
    print()
    if dp < -0.5:
        print(f'→ pass@{args.n}가 {dp:.2f}%p 하락: **능력 손실**. 학습이 정답에 도달할 경로 자체를')
        print('   지워버렸다는 뜻이며, 표본을 늘려도 복구되지 않는다. SC 스택과 근본적으로 상충.')
    elif o['top1_share'] > b['top1_share'] + 0.02:
        print(f'→ pass@{args.n}는 유지되고 1위 득표율만 상승: **분포 첨예화**.')
        print('   능력은 남아 있으나 표가 한쪽으로 쏠려 소수 정답이 묻힌다.')
    else:
        print(f'→ pass@{args.n}·분포 모두 큰 변화 없음. SC 하락은 다른 요인(노이즈 포함) 가능성.')


if __name__ == '__main__':
    main()
