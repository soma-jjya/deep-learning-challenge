"""잘림(truncation)이 정확도에서 실제로 얼마를 먹고 있는지 — GPU 불필요.

왜 보는가: 지금까지 잘림은 **집계 규칙**으로만 다뤘다(drop_trunc = 잘린 표를 버림, exp34/38).
그건 "이미 버려진 표를 어떻게 셀까"의 문제다. 정작 **max_tokens를 올려 잘림 자체를 줄이는 것**은
H1(1024→2048) 이후 한 번도 재검토하지 않았다. max_model_len이 4096이므로 2048→3072는 가능하다.

세 가지를 잰다:
  ① 잘린 표의 비율 — 낭비되는 표본의 몫
  ② 잘림이 문항 난이도와 상관되는가 — 우리가 틀리는 문항에 몰려 있으면 그만큼 회수 여지가 크다
  ③ **상한**: 잘린 표를 전부 정답으로 바꿔줬다고 가정하면 정확도가 얼마가 되나
     (실제로는 그보다 훨씬 적게 회수되지만, 이 값이 낮으면 시도 자체가 무의미하다)

사용: uv run python remote/analyze_truncation.py --samples results/val_samples.jsonl --n 32
"""
import argparse
import json
import math
from collections import Counter


def weighted_vote(ss, scale=2.0):
    """최종 스택과 동일한 확신도 가중 투표."""
    acc = {}
    for s in ss:
        a = s.get('ans')
        if a is None:
            continue
        acc[a] = acc.get(a, 0.0) + math.exp(scale * s.get('logp', 0.0))
    return max(acc, key=acc.get) if acc else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='results/val_samples.jsonl')
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples, encoding='utf-8')]
    n_tot = n_trunc = 0
    correct = 0
    trunc_on_wrong = trunc_on_right = 0
    n_wrong = n_right = 0
    ceiling = 0
    trunc_had_gold = 0

    for r in rows:
        ss = r['samples'][:args.n]
        gold = r['gold']
        t = sum(1 for s in ss if s.get('trunc'))
        n_tot += len(ss)
        n_trunc += t
        pred = weighted_vote(ss)
        ok = (pred == gold)
        correct += ok
        if ok:
            n_right += 1
            trunc_on_right += t
        else:
            n_wrong += 1
            trunc_on_wrong += t

        # 상한: 잘린 표가 하나라도 있고 현재 틀린 문항은, 잘림만 없었으면 맞혔을 수도 있다.
        # (가장 낙관적인 가정 — 실제 회수는 이보다 훨씬 적다)
        if ok or t > 0:
            ceiling += 1
        # 잘리지 않은 표 중에 정답이 이미 있었는가 = 잘림이 아니라 선택의 문제인가
        if not ok and gold in {s.get('ans') for s in ss if not s.get('trunc')}:
            trunc_had_gold += 1

    N = len(rows)
    print(f'검증 {N}문항 × n={args.n}')
    print()
    print(f'  잘린 표 비율          : {n_trunc}/{n_tot} = {n_trunc/max(1,n_tot):.1%}')
    print(f'  정답 문항의 문항당 잘림 : {trunc_on_right/max(1,n_right):.2f}표')
    print(f'  오답 문항의 문항당 잘림 : {trunc_on_wrong/max(1,n_wrong):.2f}표'
          f'   ← 이 값이 크면 잘림이 우리가 틀리는 곳에 몰려 있다')
    print()
    print(f'  현재 가중 투표        : {correct/N:.1%}  ({correct}/{N})')
    print(f'  낙관적 상한(잘린 표가 있는 오답 문항을 전부 맞혔다고 가정)'
          f' : {ceiling/N:.1%}  ({ceiling}/{N})')
    print(f'    → 회수 여지 최대 {(ceiling-correct)/N:+.2%}p ({ceiling-correct}문항)')
    print()
    print(f'  오답 중 "잘리지 않은 표에 이미 정답이 있던" 문항 : {trunc_had_gold}/{n_wrong}'
          f' = {trunc_had_gold/max(1,n_wrong):.1%}')
    print('    → 이 몫은 max_tokens를 늘려도 안 풀린다. 선택의 문제다')
    print()

    # ── 정직한 표적 집합 ──
    # max_tokens를 늘려서 회수될 **가능성이라도 있는** 문항은 세 조건을 모두 만족해야 한다:
    #   ① 지금 틀렸고 ② 잘린 표가 있고 ③ 잘리지 않은 표 어디에도 정답이 없다.
    # ③이 빠지면 "정답은 이미 손에 있었는데 못 골랐다"는 뜻이라 토큰과 무관하다.
    target = 0
    tgt_trunc_votes = 0
    for r in rows:
        ss = r['samples'][:args.n]
        gold = r['gold']
        if weighted_vote(ss) == gold:
            continue
        t = [s for s in ss if s.get('trunc')]
        if not t:
            continue
        if gold in {s.get('ans') for s in ss if not s.get('trunc')}:
            continue
        target += 1
        tgt_trunc_votes += len(t)
    print(f'  ▶ 표적 집합(틀렸고 · 잘림 있고 · 안 잘린 표엔 정답 없음) : {target}문항'
          f' = 전체의 {target/N:.2%}p')
    print(f'      이 문항들의 문항당 잘린 표 : {tgt_trunc_votes/max(1,target):.1f}개')
    print('      → max_tokens 증가로 얻을 수 있는 **절대 상한**. 잘린 풀이가 전부 정답으로')
    print('        완성된다는 가정이므로 실제 회수는 이보다 훨씬 적다.')


if __name__ == '__main__':
    main()
