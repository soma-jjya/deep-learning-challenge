"""2단계 자기수정 (H18) — 1차 풀이를 같은 모델이 검토·수정해 최종 답 결정.

규칙 적합: 모델 출력만 사용 (코드 실행·외부 도구 없음).
비교 대상: greedy 69.4% (1차=greedy이므로 순수하게 '검토 단계'의 효과 측정).
사용: uv run python remote/eval_two_pass.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.'
)
REVIEW_PROMPT = (
    'You are a strict math grader. Below is a problem and a proposed solution. '
    'Check the solution carefully for arithmetic or logical errors. '
    'If it is correct, restate the same final answer. If it has an error, solve it correctly. '
    'Either way, end with the final integer answer inside ' + BS + 'boxed{}.'
)


def main():
    from vllm import LLM, SamplingParams

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        bad_ids = set(pd.read_csv(bad)['id'])
        val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    print(f'검증 {len(val)}문항 — 1차 greedy → 2차 검토·수정')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.9, max_model_len=4096)
    tok = llm.get_tokenizer()
    greedy = SamplingParams(temperature=0, max_tokens=2048)

    # 1차: greedy 풀이
    p1 = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in val['question']]
    o1 = llm.generate(p1, greedy)
    sols = [o.outputs[0].text for o in o1]
    a1 = [extract_answer(t) for t in sols]
    acc1 = sum(1 for a, g in zip(a1, val['answer']) if a is not None and int(a) == int(g)) / len(val)
    print(f'[1차 greedy] {acc1:.1%}')

    # 2차: 검토·수정 (풀이는 앞 3000자까지만 — 컨텍스트 예산)
    p2 = [tok.apply_chat_template(
        [{'role': 'system', 'content': REVIEW_PROMPT},
         {'role': 'user', 'content': 'Problem:' + chr(10) + q + chr(10) * 2 +
          'Proposed solution:' + chr(10) + s[:3000]}],
        tokenize=False, add_generation_prompt=True)
        for q, s in zip(val['question'], sols)]
    o2 = llm.generate(p2, greedy)
    a2 = [extract_answer(o.outputs[0].text) for o in o2]
    # 2차가 무효(None)면 1차 답 유지
    final = [b if b is not None else a for a, b in zip(a1, a2)]
    acc2 = sum(1 for a, g in zip(final, val['answer']) if a is not None and int(a) == int(g)) / len(val)
    changed = sum(1 for a, b in zip(a1, a2) if b is not None and a != b)
    print(f'[2차 자기수정] {acc2:.1%} (답 변경 {changed}건)')

    os.makedirs('results', exist_ok=True)
    json.dump({'pass1_greedy': acc1, 'two_pass': acc2, 'changed': changed},
              open('results/eval_two_pass.json', 'w'), indent=2)
    print('저장: results/eval_two_pass.json')


if __name__ == '__main__':
    main()
