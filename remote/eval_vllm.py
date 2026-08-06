"""검증 세트(500, seed=123) 평가 — vLLM. greedy와 Self-Consistency 모두 지원.

사용:
    uv run python remote/eval_vllm.py --mode both                    # 베이스 모델
    uv run python remote/eval_vllm.py --mode both --adapter outputs/qlora/xxx_final
"""
import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.'
)


def majority_vote(answers):
    votes = [a for a in answers if a is not None]
    return Counter(votes).most_common(1)[0][0] if votes else 0


def weighted_vote(answer_conf_pairs):
    """확신도(평균 토큰 로그확률) 가중 다수결 (H16).

    같은 답에 투표한 샘플들의 softmax(평균 로그확률) 가중치를 합산해 최댓값 선택.
    모델 출력만 사용하므로 대회 규칙(코드 실행·외부 도구 금지)에 저촉되지 않음.
    """
    import math
    pairs = [(a, lp) for a, lp in answer_conf_pairs if a is not None]
    if not pairs:
        return 0
    m = max(lp for _, lp in pairs)
    weights = {}
    for a, lp in pairs:
        weights[a] = weights.get(a, 0.0) + math.exp((lp - m) * 2.0)  # temp 0.5 스케일
    return max(weights, key=weights.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['greedy', 'sc', 'both'], default='both')
    ap.add_argument('--adapter', default=None, help='LoRA 어댑터 경로 (없으면 베이스)')
    ap.add_argument('--n', type=int, default=8, help='SC 샘플 수')
    ap.add_argument('--val-n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(args.val_n, random_state=args.seed).reset_index(drop=True)
    # 운영진 공지(2026-08-03)의 오류 문항은 검증에서도 제외 (라벨이 틀린 문제로 채점하면 왜곡)
    bad_path = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad_path):
        bad_ids = set(pd.read_csv(bad_path)['id'])
        val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    print(f'검증 문항 {len(val)}개 (오류 문항 제외 후)')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096,
              enable_lora=args.adapter is not None, max_lora_rank=64)
    tok = llm.get_tokenizer()
    lora = LoRARequest('adapter', 1, args.adapter) if args.adapter else None

    prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in val['question']]

    results = {'adapter': args.adapter, 'val_n': args.val_n}

    def run(name, sp):
        outs = llm.generate(prompts, sp, lora_request=lora)
        preds = [majority_vote([extract_answer(c.text) for c in o.outputs]) for o in outs]
        acc = sum(int(p) == int(a) for p, a in zip(preds, val['answer'])) / len(val)
        print(f'[{name}] 정확도: {acc:.1%}')
        results[name] = acc
        # SC일 때는 같은 생성 결과로 가중 투표(H16)도 공짜로 측정
        if sp.n > 1:
            wpreds = [weighted_vote([
                (extract_answer(c.text),
                 c.cumulative_logprob / max(1, len(c.token_ids)))
                for c in o.outputs]) for o in outs]
            wacc = sum(int(p) == int(a) for p, a in zip(wpreds, val['answer'])) / len(val)
            print(f'[{name}+weighted] 확신도 가중 투표 정확도: {wacc:.1%}')
            results[name + '_weighted'] = wacc
        return preds

    if args.mode in ('greedy', 'both'):
        run('greedy', SamplingParams(temperature=0, max_tokens=2048))
    if args.mode in ('sc', 'both'):
        # logprobs=0: cumulative_logprob만 필요(가중 투표용), 토큰별 상세 리스트는 불필요 → 비용 최소화
        run(f'sc_n{args.n}', SamplingParams(n=args.n, temperature=0.7, top_p=0.8,
                                            max_tokens=2048, seed=42, logprobs=0))

    os.makedirs('results', exist_ok=True)
    tag = (args.adapter or 'base').replace('/', '_')
    out = f'results/eval_{tag}.json'
    json.dump(results, open(out, 'w'), indent=2)
    print('저장:', out)


if __name__ == '__main__':
    main()
