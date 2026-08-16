"""검증셋 표본 덤프 — 한 번 생성해두고 집계 전략을 오프라인으로 무한 실험하기 위한 재료.

지금까지는 집계 방식(다수결/가중/적응예산...)을 바꿀 때마다 GPU 생성을 다시 했다.
표본을 한 번만 만들어 저장해두면 이후 전략 탐색은 CPU에서 수 초 만에 끝난다.

저장 내용(문제당): 정답, 각 샘플의 (추출된 답, 평균 로그확률, 종료 사유, 토큰 길이)
사용: nohup uv run python remote/dump_samples.py --n 64 > dump_samples.log 2>&1 &
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
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--out', default='results/val_samples.jsonl')
    ap.add_argument('--adapter', default=None,
                    help='LoRA 어댑터 경로 (없으면 베이스). 학습이 표본 다양성에 준 영향을 '
                         '베이스 덤프와 직접 비교하기 위한 인자')
    ap.add_argument('--seed', type=int, default=42)
    # H1(1024→2048) 이후 재검토한 적 없는 축. 잘림은 전체 표의 1.7%뿐이지만 **오답 문항에
    # 12배 몰려 있고**(정답 문항 0.15표 vs 오답 문항 1.80표), "틀렸고·잘림 있고·안 잘린
    # 표엔 정답이 없는" 표적 집합이 3시드에서 28~30문항(5.8~6.2%p)이다. max_model_len이
    # 4096이므로 3072까지는 올릴 수 있다.
    ap.add_argument('--max-tokens', type=int, default=2048)
    # 확장 검증셋 지원. 483문항의 대응비교 임계가 ±1.49%p인데 우리 성공 기준이 +1.5%p라
    # 임계선에 걸쳐 판정해왔다(exp65). 문항을 늘리면 임계가 1/√N로 줄어든다.
    ap.add_argument('--val-ids', default=None,
                    help='id,answer 컬럼을 가진 CSV. 주면 기본 500표집 대신 이 목록을 쓴다')
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    if args.val_ids:
        want = pd.read_csv(args.val_ids)
        val = df[df['id'].isin(set(want['id']))].reset_index(drop=True)
        print(f'확장 검증셋 사용: {args.val_ids} ({len(val)}문항)')
    else:
        val = df.sample(500, random_state=123).reset_index(drop=True)
        bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
        if os.path.exists(bad):
            bad_ids = set(pd.read_csv(bad)['id'])
            val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    print(f'검증 {len(val)}문항 × {args.n}샘플 덤프')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096,
              enable_lora=args.adapter is not None, max_lora_rank=64)
    lora = LoRARequest('adapter', 1, args.adapter) if args.adapter else None
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in val['question']]

    outs = llm.generate(prompts, SamplingParams(
        n=args.n, temperature=args.temp, top_p=args.top_p,
        max_tokens=args.max_tokens, seed=args.seed, logprobs=0), lora_request=lora)

    os.makedirs('results', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for i, o in enumerate(outs):
            samples = []
            for c in o.outputs:
                ntok = max(1, len(c.token_ids))
                samples.append({
                    'ans': extract_answer(c.text),
                    'logp': c.cumulative_logprob / ntok,
                    'trunc': c.finish_reason == 'length',
                    'len': ntok,
                })
            f.write(json.dumps({'id': val['id'][i], 'gold': int(val['answer'][i]),
                                'samples': samples}, ensure_ascii=False) + chr(10))
    print(f'저장: {args.out} ({len(val)}문항)')
    print('다음: uv run python remote/sweep_aggregation.py')


if __name__ == '__main__':
    main()
