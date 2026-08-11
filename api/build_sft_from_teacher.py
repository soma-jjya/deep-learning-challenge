"""교사 풀이 → SFT 학습 형식 변환 (H23). GPU 불필요.

`remote/train_qlora.py`가 읽는 형식({"messages": [...]})으로 맞추고, 시스템 프롬프트를
**추론 때 쓰는 것과 정확히 동일하게** 넣는다. 학습·추론 프롬프트가 다르면 학습 효과가
프롬프트 불일치로 새어나가기 때문이다.

리플레이 혼합(선택): 어려운 문제만 학습하면 이미 잘 푸는 문제의 분포가 흔들릴 수 있다
(catastrophic forgetting). `--replay`로 기존 RFT 데이터(data/sft.jsonl)를 일부 섞으면
완화되지만, exp09b에서 외부 데이터 혼합이 최악의 결과(−2.9%p)였던 전례가 있으므로
**기본값은 혼합 없음(0)** 이며 필요할 때만 명시적으로 켠다.

사용: python api/build_sft_from_teacher.py --teacher data/teacher_full.jsonl --out data/sft_teacher.jsonl
"""
import argparse
import json
import os
import random

BS = chr(92)
SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='data/teacher_full.jsonl')
    ap.add_argument('--src', default='data/hard_problems.csv', help='question 본문 출처')
    ap.add_argument('--out', default='data/sft_teacher.jsonl')
    ap.add_argument('--replay', type=int, default=0,
                    help='기존 RFT 데이터(data/sft.jsonl)에서 섞을 샘플 수 (기본 0 = 혼합 없음)')
    ap.add_argument('--replay-src', default='data/sft.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import pandas as pd

    q = {str(r.id): str(r.question) for r in pd.read_csv(args.src).itertuples()}
    rows = []
    missing = 0
    for line in open(args.teacher, encoding='utf-8'):
        r = json.loads(line)
        if r['id'] not in q:
            missing += 1
            continue
        rows.append({'id': r['id'], 'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': q[r['id']]},
            {'role': 'assistant', 'content': r['solution']},
        ]})

    n_teacher = len(rows)
    if args.replay > 0 and os.path.exists(args.replay_src):
        pool = [json.loads(l) for l in open(args.replay_src, encoding='utf-8')]
        rng = random.Random(args.seed)
        rows.extend(rng.sample(pool, min(args.replay, len(pool))))

    random.Random(args.seed).shuffle(rows)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))

    print(f'교사 풀이 {n_teacher}개' + (f' + 리플레이 {len(rows)-n_teacher}개' if args.replay else ''))
    if missing:
        print(f'⚠️ question을 찾지 못해 제외: {missing}개')
    print(f'저장: {args.out} (총 {len(rows)}줄)')
    print()
    print('다음(서버): train_qlora.py CONFIG의 data_path를 이 파일로 바꿔 학습 →')
    print('  eval_vllm.py --mode both --n 32 로 평가.')
    print('  ⚠️ exp48 교훈: 단일 시드로 판단하지 말 것. 시드 42/43/44 대응 비교로 판정한다.')


if __name__ == '__main__':
    main()
