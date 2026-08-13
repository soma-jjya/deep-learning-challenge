"""Implicit PRM (H25) — 이미 학습해둔 DPO 어댑터를 '정책'이 아니라 '채점기'로 다시 읽는다.

근거: Yuan et al., "Free Process Rewards without Process Labels" (arXiv 2412.01981).
DPO 손실은 보상을 다음과 같이 파라미터화하고 있다:

    r(x, y) = beta * log[ P_policy(y|x) / P_ref(y|x) ]

즉 DPO를 학습시키는 순간 정책만 만든 게 아니라 **보상 모델도 함께 만든 것**이다.
게다가 이 보상은 토큰 단위로 분해되므로, 최종 답의 정오(outcome)만으로 학습했는데도
풀이 중간 과정에 대한 점수(process reward)가 공짜로 나온다. 논문은 MCTS로 스텝 라벨을
만든 Math-Shepherd(Avg 47.8)보다 이 방식(50.4)이 앞선다고 보고한다.

우리 상황에서의 의미:
  exp47에서 DPO 어댑터를 만들고 "정책 정확도가 안 올랐다"며 폐기했다. 그러나 정책으로서의
  성능과 채점기로서의 성능은 별개다. 추가 학습 없이 재해석만 하면 되는 유일한 카드다.

이전 실패들과 무엇이 다른가:
  exp18c(검증자)와 exp51(보상 헤드)은 **한 지점**(마지막 토큰/은닉상태)에서 판단을 짜냈다.
  implicit PRM은 풀이 전 구간의 토큰별 로그확률 차이를 누적한다 — 약한 신호도 수백 토큰에
  걸쳐 모으면 판별력이 생길 수 있다는 것이 가설이다.

구현 요점: PeftModel의 disable_adapter() 컨텍스트로 같은 모델에서 정책/참조를 번갈아 잰다.
모델을 두 번 올릴 필요가 없어 메모리와 시간이 절반이다.

사용:
    uv run python remote/eval_reward_bestofn.py --rm <아무경로> --n 32 --seed 42 --stage gen
    uv run python remote/eval_implicit_prm.py --dpo outputs/dpo/dpo_r8_b0.3_lr5e-06_final --gen results/rm_gen_seed42_n32.jsonl
"""
import argparse
import json
import math
import os
from collections import defaultdict

BS = chr(92)
SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')


def softmax(xs, scale=1.0):
    m = max(xs)
    e = [math.exp((x - m) * scale) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dpo', default='outputs/dpo/dpo_r8_b0.3_lr5e-06_final')
    ap.add_argument('--gen', default='results/rm_gen_seed42_n32.jsonl')
    ap.add_argument('--questions-from', default='deep-learning-challenge-2026/deep_chal_math_train.csv')
    ap.add_argument('--max-len', type=int, default=1280)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    import torch
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    rows = [json.loads(l) for l in open(args.gen, encoding='utf-8')]
    df = pd.read_csv(args.questions_from)
    df.columns = df.columns.str.strip()
    qmap = {str(r.id): str(r.question) for r in df.itertuples()}
    print(f'{len(rows)}문항 x {len(rows[0]["cands"])}후보 채점')

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B-Instruct', dtype=torch.bfloat16, device_map='cuda')
    model = PeftModel.from_pretrained(base, args.dpo).eval()

    @torch.no_grad()
    def seq_logprobs(prompts, answers):
        """각 (prompt, answer)에 대해 answer 토큰들의 로그확률 합과 개수를 돌려준다."""
        texts = [p + a for p, a in zip(prompts, answers)]
        enc = tok(texts, return_tensors='pt', padding=True, truncation=True,
                  max_length=args.max_len).to('cuda')
        plen = [len(tok(p, truncation=True, max_length=args.max_len)['input_ids'])
                for p in prompts]
        logits = model(**enc).logits.float().log_softmax(-1)
        ids, mask = enc['input_ids'], enc['attention_mask']
        out = []
        for i in range(len(texts)):
            n = int(mask[i].sum())
            s = plen[i]
            if s >= n:                      # 프롬프트가 잘려 답이 남지 않은 경우
                out.append((0.0, 1))
                continue
            # 위치 t의 토큰은 t-1의 로짓으로 예측된다
            lp = logits[i, s - 1:n - 1, :].gather(-1, ids[i, s:n].unsqueeze(-1)).sum().item()
            out.append((lp, n - s))
        return out

    total = len(rows)
    for i, r in enumerate(rows):
        q = qmap.get(str(r['id']))
        if q is None:
            for c in r['cands']:
                c['prm_sum'] = c['prm_mean'] = 0.0
            continue
        prompt = tok.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': q}],
            tokenize=False, add_generation_prompt=True)
        cands = r['cands']
        for s in range(0, len(cands), args.batch):
            part = cands[s:s + args.batch]
            ps = [prompt] * len(part)
            ans = [c['text'] for c in part]
            with model.disable_adapter():          # 참조 = 베이스 모델
                ref = seq_logprobs(ps, ans)
            pol = seq_logprobs(ps, ans)            # 정책 = DPO 어댑터 적용
            for c, (lp_p, n_p), (lp_r, _) in zip(part, pol, ref):
                c['prm_sum'] = lp_p - lp_r                     # 논문의 보상 형태(길이 편향 있음)
                c['prm_mean'] = (lp_p - lp_r) / max(1, n_p)    # 길이 정규화 변형
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{total}', flush=True)

    def pick(cands, mode):
        v = [c for c in cands if c['ans'] is not None]
        if not v:
            return None
        if mode == 'baseline':
            w = defaultdict(float)
            for c, ww in zip(v, softmax([c['logp'] for c in v], 2.0)):
                w[c['ans']] += ww
            return max(w, key=w.get)
        if mode in ('prm_sum_best', 'prm_mean_best'):
            key = 'prm_sum' if 'sum' in mode else 'prm_mean'
            return max(v, key=lambda c: c[key])['ans']
        if mode in ('prm_sum_vote', 'prm_mean_vote'):
            key = 'prm_sum' if 'sum' in mode else 'prm_mean'
            w = defaultdict(float)
            for c, ww in zip(v, softmax([c[key] for c in v], 1.0)):
                w[c['ans']] += ww
            return max(w, key=w.get)
        if mode == 'prm_mean_plus_conf':
            w = defaultdict(float)
            a = softmax([c['prm_mean'] for c in v], 1.0)
            b = softmax([c['logp'] for c in v], 2.0)
            for c, x, y in zip(v, a, b):
                w[c['ans']] += x * y
            return max(w, key=w.get)
        raise ValueError(mode)

    gold = [r['gold'] for r in rows]
    modes = ('baseline', 'prm_sum_best', 'prm_mean_best',
             'prm_sum_vote', 'prm_mean_vote', 'prm_mean_plus_conf')
    res = {}
    print()
    for m in modes:
        ok = sum(1 for r, g in zip(rows, gold) if pick(r['cands'], m) == g)
        res[m] = ok
        d = ok - res['baseline']
        print(f'[{m:20s}] {ok/len(gold):6.1%} ({ok}/{len(gold)})'
              + (f'  {d:+d}문항 ({d/len(gold)*100:+.2f}%p)' if m != 'baseline' else '  ← 기준'))

    tag = args.tag or os.path.basename(args.gen).replace('.jsonl', '')
    out = f'results/eval_implicit_prm_{tag}.json'
    json.dump({'dpo': args.dpo, 'gen': args.gen, 'total': len(gold), 'correct': res,
               'delta': {k: v - res['baseline'] for k, v in res.items()}},
              open(out, 'w'), indent=2)
    print(f'\n저장: {out}')
    print('⚠️ exp48 원칙: 시드 1개로 채택하지 말 것. 유의미하면 seed 43·44도 확인.')


if __name__ == '__main__':
    main()
