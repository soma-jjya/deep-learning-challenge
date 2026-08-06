"""NuminaMath-CoT에서 정수 답 부분집합을 뽑아 SFT 혼합 데이터를 만든다 (H12).

목적: 자기증류(자기 풀이 재학습)의 한계를 벗어나 '외부의 다양한 풀이 스타일'을 주입.
출처: AI-MO/NuminaMath-CoT (Apache 2.0, 무료 공개 — 대회 규칙 적합, README 데이터 목록 기재됨)

사용 (서버):
    uv run python remote/prep_numina.py --take 30000
→ data/numina.jsonl (외부 데이터) + data/sft_mix.jsonl (외부 + 자체 RFT 최단풀이 병합·셔플)
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--take', type=int, default=30000, help='채택할 외부 샘플 수')
    ap.add_argument('--max-solution-chars', type=int, default=2500,
                    help='너무 긴 풀이 제외 (3B에 긴 풀이는 역효과 — H14 교훈)')
    ap.add_argument('--self-data', default='data/sft_short.jsonl')
    ap.add_argument('--out-mix', default='data/sft_mix.jsonl')
    args = ap.parse_args()

    from datasets import load_dataset
    import pandas as pd

    # 검증 세트 오염 방지: 우리 검증 문항 텍스트와 겹치는 문제 제외
    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val_questions = set(df.sample(500, random_state=123)['question'].str.strip())

    ds = load_dataset('AI-MO/NuminaMath-CoT', split='train')
    print(f'NuminaMath-CoT 원본: {len(ds)}개')

    random.seed(42)
    order = list(range(len(ds)))
    random.shuffle(order)

    kept = 0
    os.makedirs('data', exist_ok=True)
    with open('data/numina.jsonl', 'w', encoding='utf-8') as f:
        for i in order:
            if kept >= args.take:
                break
            row = ds[i]
            problem, solution = row['problem'].strip(), row['solution'].strip()
            if len(solution) > args.max_solution_chars or len(solution) < 50:
                continue
            if 'boxed' not in solution:
                continue
            ans = extract_answer(solution)
            if ans is None:
                continue  # 정수 답이 아닌 문제 (증명·분수·무리수 등) 제외
            if problem in val_questions:
                continue  # 검증 오염 방지
            f.write(json.dumps({'id': f'numina-{i}', 'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': problem},
                {'role': 'assistant', 'content': solution},
            ]}, ensure_ascii=False) + chr(10))
            kept += 1
    print(f'외부 데이터 채택: {kept}개 → data/numina.jsonl')

    # 자체 RFT 데이터와 병합 + 셔플
    lines = [l for l in open('data/numina.jsonl', encoding='utf-8')]
    n_self = 0
    if os.path.exists(args.self_data):
        self_lines = [l for l in open(args.self_data, encoding='utf-8')]
        n_self = len(self_lines)
        lines += self_lines
    random.shuffle(lines)
    with open(args.out_mix, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'혼합 데이터: 외부 {kept} + 자체 {n_self} = {len(lines)}개 → {args.out_mix}')


if __name__ == '__main__':
    main()
