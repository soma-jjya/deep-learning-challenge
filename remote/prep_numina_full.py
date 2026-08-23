"""exp93: NuminaMath-CoT 전체에서 정수답 부분집합을 최대한 뽑는다 (구 exp87 재개).

prep_numina.py(exp09a, 30k 표집)와의 차이: --take 없이 필터 통과분 전부 채택,
문제 완전 중복 제거, 소스 분포 기록, 자체 데이터 혼합 없음 (순수 외부).
필터는 사전등록(prereg_exp93_bigft.md)에 고정됨.

사용 (서버):
    uv run python remote/prep_numina_full.py
→ data/numina_full.jsonl
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')

MAX_SOLUTION_CHARS = 3000
MIN_SOLUTION_CHARS = 50
MAX_PROBLEM_CHARS = 1500
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1


def main():
    from datasets import load_dataset
    import pandas as pd

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val_questions = set(df.sample(500, random_state=123)['question'].str.strip())

    ds = load_dataset('AI-MO/NuminaMath-CoT', split='train')
    print(f'NuminaMath-CoT 원본: {len(ds)}개', flush=True)

    kept = 0
    seen_problems = set()
    drop = Counter()
    sources = Counter()
    os.makedirs('data', exist_ok=True)
    with open('data/numina_full.jsonl', 'w', encoding='utf-8') as f:
        for i, row in enumerate(ds):
            if i % 100000 == 0:
                print(f'  진행 {i}/{len(ds)}, 채택 {kept}', flush=True)
            problem, solution = row['problem'].strip(), row['solution'].strip()
            if not (MIN_SOLUTION_CHARS <= len(solution) <= MAX_SOLUTION_CHARS):
                drop['solution_len'] += 1
                continue
            if len(problem) > MAX_PROBLEM_CHARS:
                drop['problem_len'] += 1
                continue
            if 'boxed' not in solution:
                drop['no_boxed'] += 1
                continue
            ans = extract_answer(solution)
            if ans is None or not (INT64_MIN <= ans <= INT64_MAX):
                drop['not_int'] += 1
                continue
            if problem in val_questions:
                drop['val_overlap'] += 1
                continue
            if problem in seen_problems:
                drop['dup'] += 1
                continue
            seen_problems.add(problem)
            sources[row.get('source', '?')] += 1
            f.write(json.dumps({'id': f'numina-{i}', 'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': problem},
                {'role': 'assistant', 'content': solution},
            ]}, ensure_ascii=False) + chr(10))
            kept += 1

    print(f'채택: {kept}개 → data/numina_full.jsonl')
    print('탈락 사유:', dict(drop))
    print('소스 분포:', dict(sources.most_common()))


if __name__ == '__main__':
    main()
