"""보상 헤드 Best-of-N 평가 (H24) — 생성 + 재순위화를 한 번에.

기존 덤프(results/val_samples*.jsonl)에는 **풀이 본문이 없어서**(ans/logp/trunc/len만 저장)
재순위화를 오프라인으로 할 수 없다. 그래서 이 스크립트는 vLLM으로 n개를 생성하면서
풀이 텍스트를 그대로 들고 있다가 보상 헤드로 점수를 매긴다.

비교 대상(같은 표본, 같은 문항 — 규칙만 다르게):
  baseline      : 확신도 가중 투표 (현재 최종 스택, 75.8%)
  rm_best       : 보상 점수 1위 풀이의 답 (순수 Best-of-N)
  rm_vote       : 보상 점수를 가중치로 쓴 투표 (softmax(reward))
  rm_plus_conf  : 보상 + 확신도 결합 투표

판정은 exp48 원칙 — 시드 여러 개로 돌려 부호가 일관될 때만 채택한다.

사용: uv run python remote/eval_reward_bestofn.py --rm outputs/reward_head/rm_r8_lr1e-05 --n 32 --seed 42
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')


def load_val(val_n=500, seed=123):
    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(val_n, random_state=seed).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        bad_ids = set(pd.read_csv(bad)['id'])
        val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    return val


def softmax(xs, scale=1.0):
    m = max(xs)
    e = [math.exp((x - m) * scale) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rm', required=True, help='보상 헤드 디렉토리 (train_reward_head.py 산출)')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rm-batch', type=int, default=8, help='보상 점수 채점 배치')
    ap.add_argument('--tag', default=None)
    # vLLM과 HF 모델을 한 프로세스에 연달아 올리면 vLLM이 잡은 GPU 메모리가 완전히
    # 반환되지 않아 OOM이 난다(del + empty_cache로도 부족). 단계를 프로세스로 분리한다.
    ap.add_argument('--stage', choices=['gen', 'score', 'both'], default='both',
                    help='gen: 생성만 하고 텍스트 저장 / score: 저장분을 채점')
    args = ap.parse_args()

    val = load_val()
    gen_path = f'results/rm_gen_seed{args.seed}_n{args.n}.jsonl'

    # ── 1단계: 생성 (풀이 텍스트 보존) ──
    if args.stage in ('gen', 'both'):
        from vllm import LLM, SamplingParams
        print(f'[gen] 검증 {len(val)}문항 x {args.n}샘플 (seed={args.seed})')
        llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
                  gpu_memory_utilization=0.85, max_model_len=4096)
        tok = llm.get_tokenizer()
        prompts = [tok.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': q}],
            tokenize=False, add_generation_prompt=True) for q in val['question']]
        outs = llm.generate(prompts, SamplingParams(
            n=args.n, temperature=args.temp, top_p=args.top_p,
            max_tokens=args.max_tokens, seed=args.seed, logprobs=0))
        os.makedirs('results', exist_ok=True)
        with open(gen_path, 'w', encoding='utf-8') as f:
            for i, o in enumerate(outs):
                cands = []
                for c in o.outputs:
                    ntok = max(1, len(c.token_ids))
                    cands.append({'text': c.text.strip()[:6000],
                                  'ans': extract_answer(c.text),
                                  'logp': c.cumulative_logprob / ntok})
                f.write(json.dumps({'id': val['id'][i], 'gold': int(val['answer'][i]),
                                    'cands': cands}, ensure_ascii=False) + chr(10))
        print(f'[gen] 저장: {gen_path}')
        if args.stage == 'gen':
            print('다음: 같은 명령에 --stage score 를 주고 별도 프로세스로 실행할 것')
            return

    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    rows = [json.loads(l) for l in open(gen_path, encoding='utf-8')]
    per_problem = [r['cands'] for r in rows]
    qmap = {i: q for i, q in enumerate(val['question'])}
    print(f'[score] {len(rows)}문항 채점 시작')

    # ── 2단계: 보상 점수 ──
    htok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    if htok.pad_token is None:
        htok.pad_token = htok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B-Instruct', dtype=torch.bfloat16, device_map='cuda')
    model = PeftModel.from_pretrained(base, args.rm).eval()
    head = nn.Linear(base.config.hidden_size, 1, dtype=torch.bfloat16).cuda()
    head.load_state_dict(torch.load(os.path.join(args.rm, 'reward_head.pt')))
    head.eval()

    @torch.no_grad()
    def rewards(question, cands):
        texts = [htok.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': question},
             {'role': 'assistant', 'content': c['text']}],
            tokenize=False, add_generation_prompt=False) for c in cands]
        out = []
        for s in range(0, len(texts), args.rm_batch):
            enc = htok(texts[s:s + args.rm_batch], return_tensors='pt',
                       padding=True, truncation=True, max_length=1536).to('cuda')
            h = model(**enc, output_hidden_states=True).hidden_states[-1]
            idx = enc['attention_mask'].sum(1) - 1
            last = h[torch.arange(h.size(0)), idx]
            out.extend(head(last).squeeze(-1).float().tolist())
        return out

    for i, cands in enumerate(per_problem):
        for c, r in zip(cands, rewards(qmap[i], cands)):
            c['rm'] = r
        if (i + 1) % 50 == 0:
            print(f'  채점 {i+1}/{len(per_problem)}', flush=True)

    # ── 3단계: 규칙별 채점 ──
    def pick(cands, mode):
        v = [c for c in cands if c['ans'] is not None]
        if not v:
            return None
        if mode == 'baseline':
            w = defaultdict(float)
            for c, ww in zip(v, softmax([c['logp'] for c in v], 2.0)):
                w[c['ans']] += ww
            return max(w, key=w.get)
        if mode == 'rm_best':
            return max(v, key=lambda c: c['rm'])['ans']
        if mode == 'rm_vote':
            w = defaultdict(float)
            for c, ww in zip(v, softmax([c['rm'] for c in v], 1.0)):
                w[c['ans']] += ww
            return max(w, key=w.get)
        if mode == 'rm_plus_conf':
            w = defaultdict(float)
            a = softmax([c['rm'] for c in v], 1.0)
            b = softmax([c['logp'] for c in v], 2.0)
            for c, x, y in zip(v, a, b):
                w[c['ans']] += x * y
            return max(w, key=w.get)
        raise ValueError(mode)

    gold = [r['gold'] for r in rows]
    res = {}
    for mode in ('baseline', 'rm_best', 'rm_vote', 'rm_plus_conf'):
        ok = sum(1 for cands, g in zip(per_problem, gold) if pick(cands, mode) == g)
        res[mode] = ok
        print(f'[{mode:14s}] {ok/len(gold):.1%} ({ok}/{len(gold)})')

    d = {k: v - res['baseline'] for k, v in res.items()}
    print()
    print('기준 대비 차이(문항): ' + ', '.join(f'{k}={v:+d}' for k, v in d.items() if k != 'baseline'))

    tag = args.tag or f'rm_seed{args.seed}'
    os.makedirs('results', exist_ok=True)
    out_path = f'results/eval_reward_{tag}.json'
    json.dump({'seed': args.seed, 'n': args.n, 'rm': args.rm,
               'total': len(gold), 'correct': res, 'delta': d},
              open(out_path, 'w'), indent=2)
    print(f'저장: {out_path}')
    print('⚠️ exp48 원칙: 시드 1개로 채택하지 말 것. seed 43·44도 돌려 부호 일관성 확인.')


if __name__ == '__main__':
    main()
