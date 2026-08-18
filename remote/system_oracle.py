"""exp73 — 실패한 시스템들이 **서로 다르게 실패했는가**. GPU 0.

## 가설
지금까지 우리는 각 실험을 "전체 정확도"로만 판정하고 낮으면 폐기했다. 그런데 병목이
선택이라면, **현재 best와 다른 문제를 맞히는 낮은 정확도 모델**이 앙상블 부품으로는
가장 값질 수 있다.

예: A=366, B=355인데 B가 A의 오답 15개를 맞히면 oracle(A,B)=381이다.
B는 단독으로는 실패했지만 라우팅 대상으로는 +15의 헤드룸을 가진 셈이다.

## 이것이 exp55(메타 선택기)와 다른 점
exp55는 **같은 n=32 안의 1위 vs 2위**를 골랐다. 여기서는 **서로 다른 정책·디코딩·학습법의
최종 예측**을 비교한다. 오류 구조가 다를 수 있다.

## 판정
union oracle이 best 대비 +8문항도 안 되면 이 차원의 헤드룸이 없는 것이고,
모델링 쪽 탐색 공간이 상당히 닫혔다고 말할 수 있다.
반대로 크게 벌어지면 라우터를 만들 가치가 생긴다(단, exp55c의 교훈 — 헤드룸이 있어도
배포 가능한 이득은 별개다).

사용: python remote/system_oracle.py
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict


def predict_all(path, n=32, scale=2.0):
    out = {}
    gold = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        c = defaultdict(float)
        for s in r['samples'][:n]:
            a = s.get('ans')
            if a is not None:
                c[a] += math.exp(scale * s.get('logp', 0.0))
        out[r['id']] = max(c, key=c.get) if c else None
        gold[r['id']] = r['gold']
    return out, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--top', type=int, default=12, help='표에 보일 시스템 수')
    args = ap.parse_args()

    paths = sorted(glob.glob('results/val_samples*.jsonl'))
    systems, gold = {}, None
    for p in paths:
        name = os.path.basename(p).replace('val_samples', '').replace('.jsonl', '') or '_base'
        try:
            pred, g = predict_all(p, args.n)
        except (OSError, json.JSONDecodeError):
            continue
        if len(pred) < 400:
            continue
        systems[name.lstrip('_')] = pred
        gold = gold or g

    ids = sorted(set.intersection(*[set(v) for v in systems.values()]))
    print(f'시스템 {len(systems)}개, 공통 {len(ids)}문항')
    print()

    acc = {k: sum(1 for i in ids if v[i] == gold[i]) for k, v in systems.items()}
    best = max(acc, key=acc.get)
    print(f'{"시스템":26s}{"정확":>7s}{"best와 다른 문항":>16s}{"best 구제":>11s}{"best 파괴":>11s}')
    print('-' * 72)
    for k in sorted(acc, key=lambda x: -acc[x])[:args.top]:
        diff = sum(1 for i in ids if systems[k][i] != systems[best][i])
        res = sum(1 for i in ids if systems[best][i] != gold[i] and systems[k][i] == gold[i])
        hrm = sum(1 for i in ids if systems[best][i] == gold[i] and systems[k][i] != gold[i])
        print(f'{k:26s}{acc[k]:7d}{diff:16d}{res:11d}{hrm:11d}')

    # 전체 시스템 중 best를 가장 잘 구제하는 것들
    print()
    print('best 구제 능력 상위 (정확도와 무관):')
    resc = sorted(((sum(1 for i in ids if systems[best][i] != gold[i] and systems[k][i] == gold[i]), k)
                   for k in systems if k != best), reverse=True)
    for r, k in resc[:8]:
        print(f'  {k:26s} +{r}문항 구제 (자체 정확도 {acc[k]})')

    # greedy union oracle
    print()
    print('greedy union oracle (완벽한 라우터를 가정한 상한):')
    chosen = [best]
    covered = {i for i in ids if systems[best][i] == gold[i]}
    print(f'  1) {best:26s} {len(covered)} ({len(covered)/len(ids):.1%})')
    for step in range(2, 7):
        gains = []
        for k in systems:
            if k in chosen:
                continue
            add = sum(1 for i in ids if i not in covered and systems[k][i] == gold[i])
            gains.append((add, k))
        gains.sort(reverse=True)
        if not gains or gains[0][0] == 0:
            break
        add, k = gains[0]
        chosen.append(k)
        covered |= {i for i in ids if systems[k][i] == gold[i]}
        print(f'  {step}) +{k:24s} {len(covered)} ({len(covered)/len(ids):.1%})  '
              f'누적 +{len(covered)-acc[best]}문항')

    print()
    print(f'전 시스템 union oracle: '
          f'{len({i for i in ids if any(v[i]==gold[i] for v in systems.values())})}'
          f' / {len(ids)}')
    print(f'현재 best({best}) = {acc[best]}')
    print()
    print('판정: union oracle이 best 대비 +8문항도 안 되면 이 차원에 헤드룸이 없다.')
    print('⚠️ 헤드룸이 있어도 배포 가능한 이득은 별개다 — exp55c에서 AUC 0.66에')
    print('   rescue 4~7배 농축이 있었지만 nested CV 배포 이득은 음수였다.')


if __name__ == '__main__':
    main()
