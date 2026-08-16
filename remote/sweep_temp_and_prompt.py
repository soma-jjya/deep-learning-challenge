"""H29·H30 — 어려운 문제용 온도 + 프롬프트를 '빼는' 방향. 한 세션에서 대조군과 함께 생성.

## H29: 조건부 온도 (어려운 문제에만 다른 온도)
분야별 진단이 보여준 것: 표를 8→64로 늘리면 정답이 후보에 들어오는 비율은 어려운 분야에서
훨씬 크게 열리는데(기하 +15.6%p, 대수 +13.8%p vs 문장제 +6.0%p) **정확도는 따라오지 않고
일부는 떨어진다**(조합론 40→30%, 대수 55→52%). 어려운 문제에서 모델이 흩뿌리는 상태라
매력적인 오답이 희귀한 정답보다 표를 빨리 모으기 때문이다.

온도는 **프롬프트를 건드리지 않고** 생성 분포를 바꾸는 유일한 수단이다. 지금까지 온도는
**전역으로만** 쓸었다(exp27: 0.4/0.7/1.0). 어려운 문제에만 다르게 준 적은 없다
(exp22는 표본 수만 바꿨고 온도는 고정이었다).

두 방향 다 그럴듯해서 둘 다 잰다:
  · t=0.3 — 흩뿌림을 줄여 한 줄기 추론을 또렷하게
  · t=1.1 — 정답이 아예 안 나오는 30% 구간을 겨냥해 더 넓게

## H30: 프롬프트를 '빼는' 방향
지금까지 프롬프트 실험 5건(exp14·20·57·58·61)은 **전부 무언가를 더한** 변형이었다.
지시를 추가하거나 예시를 붙이거나 출력을 더 요구했다. **빼는 방향은 한 번도 안 했다.**
현행 프롬프트는 exp01에서 비교 없이 정해진 뒤 그대로이고, exp58은 그것이 '5개의 추가안'보다
낫다는 것만 보였다 — 더 짧은 것과는 겨뤄본 적이 없다.

⚠️ 대조군(base @ t=0.7)을 **같은 실행 안에서** 새로 생성한다. 기존 366과 비교하면
재실행 변동(exp34b 실측 0.48%p)이 섞여 효과와 구분되지 않는다(exp58에서 확립한 설계).

사용: uv run python remote/sweep_temp_and_prompt.py --n 32 --seed 42
"""
import argparse
import json
import os
import sys

import pandas as pd

BS = chr(92)
NL = chr(10)

BASE_SYS = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')

# 형식 요구(=\boxed{})는 남긴다. 그것까지 빼면 답 추출이 무너져 정확도가 아니라
# 파서를 재는 실험이 된다.
MINIMAL_SYS = 'Solve. Put the final integer answer inside ' + BS + 'boxed{}.'

# 시스템 역할 자체가 문제인지 보는 변형 — 형식 요구를 user 쪽으로 옮긴다.
NOSYS_SUFFIX = NL * 2 + 'Put the final integer answer inside ' + BS + 'boxed{}.'

# (이름, 시스템프롬프트 or None, user 접미사, 온도)
CONFIGS = [
    ('base_t07', BASE_SYS, '', 0.7),        # 대조군 — 현행 최종 스택
    ('base_t03', BASE_SYS, '', 0.3),        # H29 집중
    ('base_t11', BASE_SYS, '', 1.1),        # H29 확산
    ('minimal_t07', MINIMAL_SYS, '', 0.7),  # H30 축소
    ('nosys_t07', None, NOSYS_SUFFIX, 0.7),  # H30 시스템 역할 제거
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--only', default=None, help='쉼표로 구분한 설정 이름')
    args = ap.parse_args()

    names = args.only.split(',') if args.only else [c[0] for c in CONFIGS]
    cfg = {c[0]: c for c in CONFIGS}
    for nm in names:
        if nm not in cfg:
            raise SystemExit(f'모르는 설정: {nm} (가능: {", ".join(cfg)})')

    sys.path.insert(0, os.path.dirname(__file__))
    from answer_extract import extract_answer

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        val = val[~val['id'].isin(set(pd.read_csv(bad)['id']))].reset_index(drop=True)
    print(f'검증 {len(val)}문항 × n={args.n}, 설정 {len(names)}종 (seed={args.seed})')

    from vllm import LLM, SamplingParams
    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()

    os.makedirs('results', exist_ok=True)
    for nm in names:
        _, system, suffix, temp = cfg[nm]
        out = f'results/val_samples_tp_{nm}_s{args.seed}.jsonl'
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            print(f'[{nm}] 이미 있음 — 건너뜀')
            continue
        prompts = []
        for q in val['question']:
            msgs = []
            if system:
                msgs.append({'role': 'system', 'content': system})
            msgs.append({'role': 'user', 'content': str(q) + suffix})
            prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True))
        print(f'[{nm}] 생성 시작 (temp={temp}, system={"있음" if system else "없음"})',
              flush=True)
        outs = llm.generate(prompts, SamplingParams(
            n=args.n, temperature=temp, top_p=args.top_p,
            max_tokens=args.max_tokens, seed=args.seed, logprobs=0))
        with open(out, 'w', encoding='utf-8') as f:
            for i, o in enumerate(outs):
                samples = []
                for c in o.outputs:
                    nt = max(1, len(c.token_ids))
                    samples.append({'ans': extract_answer(c.text),
                                    'logp': c.cumulative_logprob / nt,
                                    'trunc': c.finish_reason == 'length', 'len': nt})
                f.write(json.dumps({'id': val['id'][i], 'gold': int(val['answer'][i]),
                                    'samples': samples}, ensure_ascii=False) + NL)
        # 답 추출이 무너지지 않았는지 즉시 확인 — 프롬프트를 줄였을 때의 주된 위험이다
        rows = [json.loads(l) for l in open(out, encoding='utf-8')]
        tot = sum(len(r['samples']) for r in rows)
        na = sum(1 for r in rows for s in r['samples'] if s['ans'] is None)
        print(f'[{nm}] 저장: {out} — 답 추출 실패 {na}/{tot} = {na/tot:.2%}', flush=True)
        if na / tot > 0.05:
            print(f'⚠️ [{nm}] 추출 실패율이 5%를 넘는다 — 이 설정의 정확도는 '
                  f'프롬프트 효과가 아니라 형식 붕괴를 재는 것이다')

    print()
    print('다음: analyze_selection_gap.py로 각 덤프 채점 + '
          'analyze_conditional_temp.py로 조건부 결합')


if __name__ == '__main__':
    main()
