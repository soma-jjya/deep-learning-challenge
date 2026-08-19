"""리더보드 제출 파일 생성 — vLLM, SC 다수결. AWS 서버에서 실행.

⛔ 8/31 최종 제출에는 쓰지 말 것 (exp30 리허설에서 확인된 최대 운영 리스크).
   이 스크립트는 **체크포인트가 없어 중단되면 처음부터** 다시 돌려야 한다(최대 2.5시간 손실).
   최종전에는 반드시 2단계 경로를 쓴다:
       remote/dump_lb_samples.py         (100문항마다 저장 · 재개 가능)
       remote/make_submission_from_dump.py
       remote/validate_submission.py     (제출 직전 전수 검증)
   절차 전체는 docs/final-pipeline.md.

사용:
    uv run python remote/make_submission.py --adapter outputs/qlora/xxx_final --n 8 --tag exp06
→ results/submission_<tag>.csv (id,answer — 소문자 id, 정수만)
"""
import argparse
import os
import sys
from collections import Counter


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
    """확신도(평균 토큰 로그확률) 가중 다수결 — eval_vllm.py와 동일 로직 (exp17/25 최고 스택)."""
    import math
    pairs = [(a, lp) for a, lp in answer_conf_pairs if a is not None]
    if not pairs:
        return 0
    m = max(lp for _, lp in pairs)
    weights = {}
    for a, lp in pairs:
        weights[a] = weights.get(a, 0.0) + math.exp((lp - m) * 2.0)
    return max(weights, key=weights.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--i-know-no-checkpoint', action='store_true',
                    help='체크포인트가 없다는 것을 알고도 이 경로로 진행 (최종전 금지)')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--temp', type=float, default=0.7, help='샘플링 온도')
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--weighted', action='store_true', help='확신도 가중 투표 사용')
    ap.add_argument('--tag', required=True, help='결과 파일 이름표 (예: exp06)')
    ap.add_argument('--lb-csv', default='deep-learning-challenge-2026/deep_chal_math_leaderboard_filtered.csv')
    args = ap.parse_args()
    if not args.i_know_no_checkpoint:
        raise SystemExit(
            '⛔ 이 스크립트는 중단 시 재개가 불가능하다 (체크포인트 없음).' + chr(10) +
            '   최종 제출·장시간 실행에는 remote/dump_lb_samples.py 경로를 쓸 것.' + chr(10) +
            '   짧은 실험이라 정말 이걸 써야 하면 --i-know-no-checkpoint 를 붙일 것.')
    import pandas as pd      # 가드를 통과한 뒤에 무겁게 로드한다

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lb = pd.read_csv(args.lb_csv)
    lb.columns = lb.columns.str.strip()
    print(f'리더보드 {len(lb)}문항 ({args.lb_csv})')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.92,
              enable_lora=args.adapter is not None, max_lora_rank=64)
    tok = llm.get_tokenizer()
    lora = LoRARequest('adapter', 1, args.adapter) if args.adapter else None

    prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in lb['question']]

    sp = SamplingParams(n=args.n, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=42,
                        logprobs=0 if args.weighted else None)
    # 청크 처리 — n이 클 때 결과가 통째로 RAM에 쌓여 OOM으로 죽는 것 방지 (exp29 사망 원인, 2026-08-10)
    CHUNK = 100
    preds = []
    for s in range(0, len(prompts), CHUNK):
        outs = llm.generate(prompts[s:s + CHUNK], sp, lora_request=lora)
        if args.weighted:
            preds += [weighted_vote([
                (extract_answer(c.text), c.cumulative_logprob / max(1, len(c.token_ids)))
                for c in o.outputs]) for o in outs]
        else:
            preds += [majority_vote([extract_answer(c.text) for c in o.outputs]) for o in outs]
        del outs
        print(f'  진행 {len(preds)}/{len(prompts)}문항')

    os.makedirs('results', exist_ok=True)
    out_path = f'results/submission_{args.tag}.csv'
    sub = pd.DataFrame({'id': lb['id'], 'answer': preds})
    sub['answer'] = sub['answer'].astype('int64')
    sub.to_csv(out_path, index=False)
    print(f'저장: {out_path} ({len(sub)}행). 빈 답 없음 확인: {sub["answer"].notna().all()}')


if __name__ == '__main__':
    main()
