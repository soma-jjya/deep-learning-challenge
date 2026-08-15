"""토큰 단위 로그확률까지 저장하는 덤프 — 확신도 계열 기법을 GPU 없이 검증하기 위한 재료.

왜 필요한가: 기존 덤프(`val_samples*.jsonl`)는 풀이당 **평균** logprob만 저장한다. 그런데
프롬프트를 바꾸지 않는 기법들(구간별 확신도, 엔트로피 기반 가중, 최악 국소 구간 탐지 등)은
전부 **토큰 단위** 신호를 요구한다. 한 번 떠두면 그 계열 전체가 CPU 실험이 된다.

우리 제약과의 정합: 프롬프트를 그대로 쓴다(exp58에서 프롬프트 변경은 5종 전부 하락).
모델·샘플링 설정도 최종 스택과 동일하다. 오직 **무엇을 기록하는가**만 다르다.

용량: 483문항 × 32샘플 × 평균 518토큰 ≈ 800만 값. 소수 3자리로 반올림해 저장한다
(확신도 비교에 그 이상의 정밀도는 불필요하고, 파일이 4배 커지면 CPU 실험이 느려진다).

사용: uv run python remote/dump_token_logprobs.py --n 32 --seed 42
"""
import argparse
import json
import os
import sys

import pandas as pd

BS = chr(92)
SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from answer_extract import extract_answer

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        val = val[~val['id'].isin(set(pd.read_csv(bad)['id']))].reset_index(drop=True)
    print(f'검증 {len(val)}문항 × {args.n}샘플, 토큰 단위 logprob 저장 (seed={args.seed})')

    from vllm import LLM, SamplingParams
    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True) for q in val['question']]

    # logprobs=0 → 선택된 토큰의 logprob만 반환(상위 후보 목록은 불필요하고 용량만 먹는다).
    # ⚠️ 이 인자를 빠뜨리면 vLLM이 logprob을 전부 None으로 준다 — exp37이 그 버그로
    #    0/65건을 호출해놓고 '가설 기각'으로 기록될 뻔했다.
    outs = llm.generate(prompts, SamplingParams(
        n=args.n, temperature=args.temp, top_p=args.top_p,
        max_tokens=args.max_tokens, seed=args.seed, logprobs=0))

    out = args.out or f'results/val_tok_s{args.seed}.jsonl'
    os.makedirs('results', exist_ok=True)
    n_none = 0
    with open(out, 'w', encoding='utf-8') as f:
        for i, o in enumerate(outs):
            samples = []
            for c in o.outputs:
                lps = []
                for pos, tid in zip(c.logprobs or [], c.token_ids):
                    if not pos or tid not in pos:
                        n_none += 1
                        continue
                    lps.append(round(pos[tid].logprob, 3))
                nt = max(1, len(c.token_ids))
                samples.append({'ans': extract_answer(c.text),
                                'logp': c.cumulative_logprob / nt,
                                'trunc': c.finish_reason == 'length',
                                'len': nt, 'tlp': lps})
            f.write(json.dumps({'id': val['id'][i], 'gold': int(val['answer'][i]),
                                'samples': samples}, ensure_ascii=False) + chr(10))

    size = os.path.getsize(out) / 2 ** 20
    print(f'저장: {out} ({size:.0f} MiB)')
    if n_none:
        print(f'⚠️ logprob을 못 읽은 토큰 {n_none}개 — 0이 아니면 원인을 먼저 볼 것')
    # 무결성 확인: 토큰 수와 저장된 logprob 수가 맞는가
    rows = [json.loads(l) for l in open(out, encoding='utf-8')]
    bad_len = sum(1 for r in rows for s in r['samples'] if len(s['tlp']) != s['len'])
    tot = sum(len(r['samples']) for r in rows)
    print(f'무결성: 토큰 수 불일치 {bad_len}/{tot}개 샘플')
    print('다음: 이 파일로 확신도 계열 기법을 CPU에서 검증한다(재생성 불필요).')


if __name__ == '__main__':
    main()
