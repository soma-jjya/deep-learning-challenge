"""서로 다른 정책(베이스 / 교사 어댑터)의 표본을 합쳐 투표한다 — GPU 불필요.

왜: exp53c에서 교사 어댑터는 **pass@32가 베이스보다 높았다**(평균 +1.7문항). 그런데 그
어댑터로 표본을 통째로 갈아치우면 최종 점수는 −8.7문항이었다. 갈아치우는 대신 **섞으면**
베이스의 투표 성향은 유지하면서 후보 풀만 넓힐 수 있는지 확인한다.

exp19(멀티 어댑터 앙상블)가 이미 실패했지만, 그때 쓴 어댑터들은 pass@n이 베이스보다 **낮았다**.
pass@n이 더 높은 정책을 섞어본 것은 이번이 처음이다.

사용: uv run python remote/merge_policy_vote.py --a results/val_samples.jsonl \
        --b results/val_samples_teacher_full_s42.jsonl --n 32
"""
import argparse
import json
import math


def weighted_vote(ss, scale=2.0):
    acc = {}
    for s in ss:
        a = s.get('ans')
        if a is None:
            continue
        acc[a] = acc.get(a, 0.0) + math.exp(scale * s.get('logp', 0.0))
    return max(acc, key=acc.get) if acc else None


def load(path):
    return {json.loads(l)['id']: json.loads(l) for l in open(path, encoding='utf-8')}


def acc_of(rows, pick):
    ok = 0
    for r in rows.values():
        if weighted_vote(pick(r)) == r['gold']:
            ok += 1
    return ok, len(rows)


def passn(samples, gold):
    return gold in {s.get('ans') for s in samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='정책 A 덤프 (보통 베이스)')
    ap.add_argument('--b', required=True, help='정책 B 덤프 (보통 어댑터)')
    ap.add_argument('--n', type=int, default=32, help='최종 표 수 (섞을 때 A/B 절반씩)')
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    ids = sorted(set(A) & set(B))
    print(f'공통 {len(ids)}문항, n={args.n}')
    if not ids:
        raise SystemExit('공통 문항이 없다 — 덤프가 서로 다른 검증셋이다')

    half = args.n // 2
    rows = {i: A[i] for i in ids}

    variants = {
        f'A만 (n={args.n})': lambda r: A[r['id']]['samples'][:args.n],
        f'B만 (n={args.n})': lambda r: B[r['id']]['samples'][:args.n],
        f'A{half}+B{half} 혼합': lambda r: (A[r['id']]['samples'][:half]
                                          + B[r['id']]['samples'][:half]),
        f'A{args.n}+B{args.n} 전량합집합': lambda r: (A[r['id']]['samples'][:args.n]
                                                  + B[r['id']]['samples'][:args.n]),
    }
    print()
    base_ok = None
    for name, pick in variants.items():
        ok, n = acc_of(rows, pick)
        if base_ok is None:
            base_ok = ok
        d = ok - base_ok
        print(f'  {name:24s} {ok/n:6.1%}  ({ok}/{n})  {d:+d}문항')

    # pass@n 도 함께 — 후보 풀이 실제로 넓어졌는지 확인
    print()
    pa = sum(passn(A[i]['samples'][:args.n], A[i]['gold']) for i in ids)
    pb = sum(passn(B[i]['samples'][:args.n], A[i]['gold']) for i in ids)
    pu = sum(passn(A[i]['samples'][:args.n] + B[i]['samples'][:args.n], A[i]['gold'])
             for i in ids)
    n = len(ids)
    print(f'  pass@{args.n} A        : {pa/n:6.1%} ({pa})')
    print(f'  pass@{args.n} B        : {pb/n:6.1%} ({pb})')
    print(f'  pass@{2*args.n} 합집합  : {pu/n:6.1%} ({pu})   ← 후보 풀은 이만큼 넓어진다')
    print()
    print('  합집합 pass@n이 크게 늘었는데 투표 정확도가 안 따라오면,')
    print('  그것은 다시 한 번 "후보가 아니라 선택이 병목"이라는 증거다.')


if __name__ == '__main__':
    main()
