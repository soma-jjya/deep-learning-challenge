"""베이스 모델이 못 푸는 문제 목록 추출 — 상용 API 교사 증류의 대상 선정 (GPU 불필요).

왜 이 부분집합인가:
RFT(exp04)는 베이스가 스스로 정답에 도달한 풀이만 채택하므로, **못 푸는 문제는 학습
데이터에 존재조차 하지 않는다**(exp04 실측: 16,500문제 중 정답 도달 13,190개=79.9%).
학습 6전이 전부 실패한 원인 중 하나로 지목된 '데이터 편향'이 바로 이것이다.
상용 API 교사는 정확히 이 빈칸 — RFT가 구조적으로 만들 수 없는 데이터 — 를 채운다.

규칙 근거: prd.md:30 "학습 데이터 구축 목적의 상용 API 사용은 허용 (예: GPT-4로 풀이 생성)".
prd.md:31의 금지 사항(테스트 문제 답 직접 생성)에 해당하지 않도록 **train 문제만** 대상으로 한다.

사용: uv run python remote/export_hard_problems.py
출력: data/hard_problems.csv  (id, question, answer)
"""
import argparse
import json
import os

import pandas as pd

TRAIN = 'deep-learning-challenge-2026/deep_chal_math_train.csv'
BAD = 'deep-learning-challenge-2026/train_filtered_ids.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rft-progress', default='data/rft_progress.json',
                    help='RFT가 처리한 문제 목록 (done_ids)')
    ap.add_argument('--sft', default='data/sft.jsonl',
                    help='RFT 채택 풀이 — 여기 등장하는 id = 베이스가 푼 문제')
    ap.add_argument('--out', default='data/hard_problems.csv')
    ap.add_argument('--val-seed', type=int, default=123, help='검증셋 시드 — 절대 변경 금지')
    ap.add_argument('--val-n', type=int, default=500)
    args = ap.parse_args()

    df = pd.read_csv(TRAIN)
    df.columns = df.columns.str.strip()

    # 검증셋과 오류 문항은 학습 데이터에서 반드시 제외
    val_ids = set(df.sample(args.val_n, random_state=args.val_seed)['id'])
    bad_ids = set(pd.read_csv(BAD)['id']) if os.path.exists(BAD) else set()

    # RFT가 실제로 처리한 문제만 '풀었다/못 풀었다'를 말할 수 있다.
    # 미처리 문제를 '못 푸는 문제'로 넣으면 근거 없는 데이터가 섞인다.
    processed = set()
    if os.path.exists(args.rft_progress):
        processed = set(json.load(open(args.rft_progress))['done_ids'])
    else:
        raise SystemExit(f'RFT 진행 파일이 없습니다: {args.rft_progress}')

    solved = set()
    with open(args.sft, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if 'id' in r:
                solved.add(r['id'])

    hard_ids = processed - solved - val_ids - bad_ids
    hard = df[df['id'].isin(hard_ids)][['id', 'question', 'answer']].reset_index(drop=True)

    os.makedirs('data', exist_ok=True)
    hard.to_csv(args.out, index=False)

    print(f'train 전체            : {len(df)}')
    print(f'RFT 처리 완료          : {len(processed)}')
    print(f'  그중 베이스가 푼 문제  : {len(processed & solved)}')
    print(f'  그중 못 푼 문제        : {len(processed - solved)}')
    print(f'검증셋·오류문항 제외 후  : {len(hard)}  → {args.out}')
    print()
    print('다음: 로컬에서 api/gen_teacher_solutions.py 실행 (API 키 필요, GPU 불필요)')


if __name__ == '__main__':
    main()
