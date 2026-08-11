"""DPO 선호쌍 생성 (H15) — data/verifier.jsonl을 (문제, 정답풀이, 오답풀이) 쌍으로 변환.

검증자 학습용으로 이미 만들어둔 6000문제×4샘플 라벨 데이터를 재활용한다(추가 생성 0).
같은 문제 안에서 정답 풀이 하나와 오답 풀이 하나를 짝지어 선호쌍을 만든다.

SFT가 5전 전패한 이유는 분포를 정답 하나로 수축시켜 SC 다양성을 죽였기 때문이다(exp09b:
greedy -1.9%p보다 SC -2.9%p가 더 나빴다 = 다양성 손실이 정확도 손실보다 컸다).
DPO는 정답 쪽을 끌어올리는 게 아니라 오답 쪽을 상대적으로 내리는 학습이라 그 경로를
우회할 여지가 있다 — 이것이 이 실험의 유일한 근거이자, 동시에 검증해야 할 가정이다.

사용: uv run python remote/prep_dpo_data.py --max-pairs-per-problem 2
출력: data/dpo.jsonl  {"id", "question", "chosen", "rejected"}
"""
import argparse
import json
import os
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/verifier.jsonl')
    ap.add_argument('--out', default='data/dpo.jsonl')
    ap.add_argument('--max-pairs-per-problem', type=int, default=2,
                    help='쉬운 문제(정답 풀이가 많은)가 데이터를 독식하지 않게 상한을 둔다')
    ap.add_argument('--min-len', type=int, default=50,
                    help='너무 짧은 풀이는 추출 실패·잘림일 가능성이 커서 제외')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_id = defaultdict(lambda: {'q': None, 'pos': [], 'neg': []})
    total = 0
    for line in open(args.src, encoding='utf-8'):
        r = json.loads(line)
        total += 1
        sol = r['solution'].strip()
        if len(sol) < args.min_len:
            continue
        slot = by_id[r['id']]
        slot['q'] = r['question']
        (slot['pos'] if r['label'] else slot['neg']).append(sol)

    pairs = []
    both = 0
    for pid, slot in by_id.items():
        if not slot['pos'] or not slot['neg']:
            continue
        both += 1
        # 정답 쪽은 짧은 순(간결·완결된 풀이일 확률이 높다), 오답 쪽은 무작위로 뽑아
        # 특정 오류 유형에만 과적합되지 않게 한다.
        pos = sorted(slot['pos'], key=len)
        neg = slot['neg'][:]
        rng.shuffle(neg)
        for k in range(min(args.max_pairs_per_problem, len(pos), len(neg))):
            pairs.append({'id': pid, 'question': slot['q'],
                          'chosen': pos[k], 'rejected': neg[k]})

    rng.shuffle(pairs)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + chr(10))

    print(f'원본 {total}줄 / 문제 {len(by_id)}개')
    print(f'정답·오답 둘 다 있는 문제 {both}개 (선호쌍 생성 가능)')
    print(f'생성된 선호쌍 {len(pairs)}개 → {args.out}')


if __name__ == '__main__':
    main()
