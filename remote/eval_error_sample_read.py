"""exp16 보조 스크립트 — 오답 30개의 전체 풀이 텍스트를 확보해 유형 분류 근거로 삼는다.

results/wrong_analysis.jsonl의 앞 30건에 대해 greedy(temp0) 1개 샘플의 전체 텍스트를
results/wrong_sample_texts.jsonl에 저장. 사람이 직접 읽고 유형 분류(EXPERIMENTS.md).

사용: uv run python remote/eval_error_sample_read.py
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


def main():
    from vllm import LLM, SamplingParams

    wrong = [json.loads(l) for l in open('results/wrong_analysis.jsonl')][:30]
    ids = [r['id'] for r in wrong]

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    df = df.set_index('id')
    questions = [df.loc[i, 'question'] for i in ids]

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()

    prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in questions]

    sp = SamplingParams(n=1, temperature=0.0, max_tokens=2048)
    outs = llm.generate(prompts, sp)

    os.makedirs('results', exist_ok=True)
    with open('results/wrong_sample_texts.jsonl', 'w') as f:
        for r, q, o in zip(wrong, questions, outs):
            text = o.outputs[0].text
            f.write(json.dumps({
                'id': r['id'],
                'question': q,
                'answer': r['answer'],
                'extracted_answers_sc8': r['extracted_answers'],
                'greedy_text': text,
                'greedy_extracted': extract_answer(text),
            }, ensure_ascii=False) + '\n')
    print(f'저장: results/wrong_sample_texts.jsonl ({len(wrong)}건)')


if __name__ == '__main__':
    main()
