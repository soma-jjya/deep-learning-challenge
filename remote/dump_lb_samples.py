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
    ap.add_argument('--seed', type=int, default=42, help='vLLM 샘플링 시드 (분산 측정용 다중 표본 세트 생성 시 변경)')
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--chunk', type=int, default=100, help='이 단위로 생성·저장 (OOM 방지·재개 지원)')
    ap.add_argument('--lb-csv', default='deep-learning-challenge-2026/deep_chal_math_leaderboard_filtered.csv')
    ap.add_argument('--out', default='results/lb_samples.jsonl')
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    lb = pd.read_csv(args.lb_csv)
    lb.columns = lb.columns.str.strip()
    print(f'리더보드 {len(lb)}문항 × {args.n}샘플 덤프 시작 (seed={args.seed})')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()

    # ── 청크 처리 + 중간 저장 (2026-08-10) ──
    # 831문항 × 96샘플을 한 번에 생성하면 결과가 통째로 RAM에 쌓여 OOM으로 죽는다
    # (실제로 exp29가 이 방식으로 사망). 청크마다 디스크에 쓰고 메모리를 비우면
    # 사용량이 1/N로 줄고, 중간에 죽어도 done 목록을 보고 이어서 재개된다.
    os.makedirs('results', exist_ok=True)
    prog = args.out + '.progress'
    done = set()
    if os.path.exists(prog) and os.path.exists(args.out):
        done = set(json.load(open(prog))['done'])
        print(f'이어서 진행: {len(done)}문항 완료됨')

    todo = [(i, q) for i, q in enumerate(lb['question']) if lb['id'][i] not in done]
    sp = SamplingParams(n=args.n, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed, logprobs=0)

    for s in range(0, len(todo), args.chunk):
        part = todo[s:s + args.chunk]
        prompts = [tok.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': q}],
            tokenize=False, add_generation_prompt=True) for _, q in part]
        outs = llm.generate(prompts, sp)
        with open(args.out, 'a', encoding='utf-8') as f:
            for (i, _), o in zip(part, outs):
                samples = []
                for c in o.outputs:
                    ntok = max(1, len(c.token_ids))
                    samples.append({'ans': extract_answer(c.text),
                                    'logp': c.cumulative_logprob / ntok,
                                    'trunc': c.finish_reason == 'length',
                                    'len': ntok})
                f.write(json.dumps({'id': lb['id'][i], 'samples': samples},
                                   ensure_ascii=False) + chr(10))
                done.add(lb['id'][i])
        json.dump({'done': sorted(done)}, open(prog, 'w'))
        del outs
        print(f'진행 {len(done)}/{len(lb)}문항 저장됨')
    print(f'저장: {args.out} ({len(lb)}문항 × {args.n})')
    print('다음: remote/make_submission_from_dump.py 로 원하는 만큼 제출 파일 생성 (GPU 불필요)')


if __name__ == '__main__':
    main()
