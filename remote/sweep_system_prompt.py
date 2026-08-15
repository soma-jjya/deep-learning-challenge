"""H27 — 시스템 프롬프트 단일 변형 스윕. 우리 최고 스택은 그대로 두고 조건화만 바꾼다.

왜 지금까지 안 했나: 프롬프트 문자열은 **exp01에서 정해진 뒤 57개 실험 동안 그대로**였다.
exp14가 4종을 만들긴 했으나 **섞어서 앙상블로만** 썼고(각 2샘플=8표), 각 프롬프트의 개별
SC 성능은 한 번도 재지 않았다. "더 나은 단일 프롬프트가 있는가"는 미답 상태다.

왜 유망한가: 학습 7전과 퓨샷 2전이 실패한 이유가 "이미 조율된 모델을 건드리면 손해"라면,
**모델을 건드리지 않고 지시만 바꾸는 것**이 남은 유일한 생성 측 개입이다.

효율: 모델을 한 번만 올리고 프롬프트별로 생성한다(변형마다 재적재하면 5분씩 낭비).

사용: uv run python remote/sweep_system_prompt.py --n 32 --seed 42
"""
import argparse
import json
import os
import sys

import pandas as pd

BS = chr(92)
NL = chr(10)
BOX = BS + 'boxed{}'

PROMPTS = {
    # 대조군 — 현재 최종 스택이 쓰는 문자열 그대로. 반드시 같은 실행 안에 둔다
    # (다른 실행과 비교하면 vLLM 비결정성이 섞여 ±1%p가 붙는다).
    'base': (
        'You are an expert competition mathematician. '
        'Solve the problem step by step. '
        'The final answer is always an integer. '
        'Put your final integer answer inside ' + BOX + '.'),

    # 검산 강조 — 오답의 10%가 계산 실수였다(exp16). 마지막에 되짚게 한다.
    'verify': (
        'You are an expert competition mathematician. '
        'Solve the problem step by step, keeping every arithmetic step explicit. '
        'Before giving your final answer, re-check your arithmetic and confirm that the '
        'answer satisfies every condition stated in the problem. '
        'The final answer is always an integer. '
        'Put your final integer answer inside ' + BOX + '.'),

    # 유형 식별 우선 — 분야별 진단에서 조합론 30%, 정수론·기하가 약했다. 접근을 먼저
    # 고르게 하면 그 유형의 표준 기법으로 들어갈 확률이 오를 수 있다.
    'classify': (
        'You are an expert competition mathematician. '
        'First identify what kind of problem this is and which standard technique applies. '
        'Then carry out that technique step by step. '
        'The final answer is always an integer. '
        'Put your final integer answer inside ' + BOX + '.'),

    # 직진 — 오답 문항의 평균 풀이 길이가 843, 정답 문항은 411이었다(exp50). 헤맬수록
    # 길어진다면, 에두르지 말라는 지시가 도움이 될 수 있다.
    # ⚠️ 이것은 상관을 인과로 읽은 가설이다(exp55에서 같은 함정에 빠졌다). 그래서
    #    '검증 대상'으로 넣는 것이지 기대치가 높아서가 아니다.
    'direct': (
        'You are an expert competition mathematician. '
        'Solve the problem directly and efficiently. Choose the shortest correct path and '
        'do not explore dead ends or restate the problem. '
        'The final answer is always an integer. '
        'Put your final integer answer inside ' + BOX + '.'),

    # 조건 정리 우선 — 문장제·경시형 모두 조건 누락이 접근 오류(67%)의 흔한 원인이다.
    'extract': (
        'You are an expert competition mathematician. '
        'Begin by listing every given quantity and constraint from the problem in your own '
        'words, then solve step by step using all of them. '
        'The final answer is always an integer. '
        'Put your final integer answer inside ' + BOX + '.'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--only', default=None, help='쉼표로 구분한 프롬프트 이름 (기본 전부)')
    args = ap.parse_args()

    names = args.only.split(',') if args.only else list(PROMPTS)
    for nm in names:
        if nm not in PROMPTS:
            raise SystemExit(f'모르는 프롬프트: {nm} (가능: {", ".join(PROMPTS)})')

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        val = val[~val['id'].isin(set(pd.read_csv(bad)['id']))].reset_index(drop=True)
    print(f'검증 {len(val)}문항 × n={args.n}, 프롬프트 {len(names)}종 (seed={args.seed})')

    from vllm import LLM, SamplingParams
    sys.path.insert(0, os.path.dirname(__file__))
    from answer_extract import extract_answer

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=args.n, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed, logprobs=0)

    os.makedirs('results', exist_ok=True)
    for nm in names:
        out = f'results/val_samples_sp_{nm}_s{args.seed}.jsonl'
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            print(f'[{nm}] 이미 있음 — 건너뜀')
            continue
        prompts = [tok.apply_chat_template(
            [{'role': 'system', 'content': PROMPTS[nm]},
             {'role': 'user', 'content': q}],
            tokenize=False, add_generation_prompt=True) for q in val['question']]
        print(f'[{nm}] 생성 시작', flush=True)
        outs = llm.generate(prompts, sp)
        with open(out, 'w', encoding='utf-8') as f:
            for i, o in enumerate(outs):
                samples = []
                for c in o.outputs:
                    nt = max(1, len(c.token_ids))
                    samples.append({'ans': extract_answer(c.text),
                                    'logp': c.cumulative_logprob / nt,
                                    'trunc': c.finish_reason == 'length',
                                    'len': nt})
                f.write(json.dumps({'id': val['id'][i], 'gold': int(val['answer'][i]),
                                    'samples': samples}, ensure_ascii=False) + chr(10))
        print(f'[{nm}] 저장: {out}', flush=True)

    print()
    print('다음: 각 덤프를 analyze_selection_gap.py로 채점해 base와 비교.')
    print('⚠️ base도 이 실행 안에서 새로 생성했다 — 다른 실행의 75.8%와 비교하면')
    print('   vLLM 비결정성이 섞인다. 반드시 같은 실행의 base를 기준으로 볼 것.')


if __name__ == '__main__':
    main()
