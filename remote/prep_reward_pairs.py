"""보상 헤드용 쌍대 데이터 생성 (H24) — data/verifier.jsonl 재활용, 추가 생성 0.

DPO 쌍(prep_dpo_data.py)과의 차이: DPO는 문제당 상한 2쌍으로 제한했지만(정책이 특정
문제에 과적합되는 것을 막기 위해), 보상 헤드는 **순위 판별력** 자체를 배우는 것이 목표라
한 문제 안의 교차쌍을 더 많이 쓰는 편이 유리하다. 기본값은 문제당 최대 4쌍.

사용: uv run python remote/prep_reward_pairs.py
출력: data/reward_pairs.jsonl  {"id","question","chosen","rejected"}
"""
import argparse
import json
import os
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/verifier.jsonl')
    ap.add_argument('--out', default='data/reward_pairs.jsonl')
    ap.add_argument('--max-pairs', type=int, default=4, help='문제당 최대 쌍 수')
    ap.add_argument('--min-len', type=int, default=50)
    ap.add_argument('--holdout', type=float, default=0.05,
                    help='쌍 정확도를 정직하게 재기 위한 검증용 분리 비율')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by = defaultdict(lambda: {'q': None, 'pos': [], 'neg': []})
    for line in open(args.src, encoding='utf-8'):
        r = json.loads(line)
        sol = r['solution'].strip()
        if len(sol) < args.min_len:
            continue
        slot = by[r['id']]
        slot['q'] = r['question']
        (slot['pos'] if r['label'] else slot['neg']).append(sol)

    pairs = []
    usable = 0
    for pid, s in by.items():
        if not s['pos'] or not s['neg']:
            continue
        usable += 1
        cross = [(p, n) for p in s['pos'] for n in s['neg']]
        rng.shuffle(cross)
        for p, n in cross[:args.max_pairs]:
            pairs.append({'id': pid, 'question': s['q'], 'chosen': p, 'rejected': n})

    rng.shuffle(pairs)
    n_hold = int(len(pairs) * args.holdout)
    hold, train = pairs[:n_hold], pairs[n_hold:]

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    for path, rows in ((args.out, train), (args.out.replace('.jsonl', '_holdout.jsonl'), hold)):
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + chr(10))

    print(f'문제 {len(by)}개 중 정답·오답 둘 다 있는 문제 {usable}개')
    print(f'학습 쌍 {len(train)}개 → {args.out}')
    print(f'검증 쌍 {len(hold)}개 → {args.out.replace(".jsonl", "_holdout.jsonl")}')
    print()
    print('다음: nohup uv run python remote/train_reward_head.py > train_rm.log 2>&1 &')


if __name__ == '__main__':
    main()
