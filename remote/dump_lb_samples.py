"""리더보드 표본 대량 덤프 — 한 번 생성해두고 제출 파일을 무한히 찍어내기 위한 재료.

지금까지는 제출 후보 1개당 GPU 1~2시간을 썼다(그래서 하루 1~2개가 한계).
표본을 한 번(n=96) 만들어 저장하면, 이후 어떤 n·어떤 집계 규칙의 제출 파일도
GPU 없이 수 초 만에 만들 수 있다 → 하루 최대 건수 제출이 가능해진다.

사용: nohup uv run python remote/dump_lb_samples.py --n 96 > dump_lb.log 2>&1 &
출력: results/lb_samples.jsonl  (용량 커서 커밋 금지 — 서버에만 보관)
"""
import argparse
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
    'Put your final integer answer inside ' + BS + 'boxed{}.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=96)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--lb-csv', default='deep-learning-challenge-2026/deep_chal_math_leaderboard_filtered.csv')
    ap.add_argument('--out', default='results/lb_samples.jsonl')
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    lb = pd.read_csv(args.lb_csv)
    lb.columns = lb.columns.str.strip()
    print(f'리더보드 {len(lb)}문항 × {args.n}샘플 덤프 시작')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in lb['question']]

    outs = llm.generate(prompts, SamplingParams(
        n=args.n, temperature=args.temp, top_p=args.top_p,
        max_tokens=args.max_tokens, seed=42, logprobs=0))

    os.makedirs('results', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for i, o in enumerate(outs):
            samples = []
            for c in o.outputs:
                ntok = max(1, len(c.token_ids))
                samples.append({'ans': extract_answer(c.text),
                                'logp': c.cumulative_logprob / ntok,
                                'trunc': c.finish_reason == 'length',
                                'len': ntok})
            f.write(json.dumps({'id': lb['id'][i], 'samples': samples},
                               ensure_ascii=False) + chr(10))
    print(f'저장: {args.out} ({len(lb)}문항 × {args.n})')
    print('다음: remote/make_submission_from_dump.py 로 원하는 만큼 제출 파일 생성 (GPU 불필요)')


if __name__ == '__main__':
    main()
