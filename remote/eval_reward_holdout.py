"""보상 헤드 holdout 평가 — 학습 중 지표의 낙관 편향을 걷어낸다. 수 분 소요.

학습 로그의 "쌍 정확도"는 **학습에 쓴 쌍**에서 잰 값이라 낙관적이고, 160쌍 창으로
집계돼 표본오차가 ±0.075(2σ)로 크다. 판정은 학습에 한 번도 쓰지 않은 holdout에서
한 번에 재야 한다(prep_reward_pairs.py가 5%를 미리 떼어놓았다).

이 값이 판정 기준이다:
  ~0.50  표현에도 정오 신호 없음 → 선택 축 최종 폐쇄
  ~0.60  exp47 DPO(0.60)·exp18c와 동급 — Best-of-N으로 회수 어려움
  0.70+  표현에는 신호가 있다 → 재순위화 평가로 진행할 가치 있음

사용: uv run python remote/eval_reward_holdout.py --rm outputs/reward_head/<RUN>
"""
import argparse
import json
import math
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rm', required=True)
    ap.add_argument('--data', default='data/reward_pairs_holdout.jsonl')
    ap.add_argument('--max-len', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    from peft import PeftModel

    BS = chr(92)
    SYSTEM_PROMPT = (
        'You are an expert competition mathematician. '
        'Solve the problem step by step. '
        'The final answer is always an integer. '
        'Put your final integer answer inside ' + BS + 'boxed{}.')

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(
        'Qwen/Qwen2.5-3B-Instruct', dtype=torch.bfloat16, device_map='cuda',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True))
    model = PeftModel.from_pretrained(base, args.rm).eval()
    head = nn.Linear(base.config.hidden_size, 1, dtype=torch.bfloat16).cuda()
    head.load_state_dict(torch.load(os.path.join(args.rm, 'reward_head.pt')))
    head.eval()

    rows = [json.loads(l) for l in open(args.data, encoding='utf-8')]
    print(f'holdout {len(rows)}쌍 채점 (학습에 한 번도 쓰지 않은 데이터)')

    @torch.no_grad()
    def score(texts):
        out = []
        for s in range(0, len(texts), args.batch):
            enc = tok(texts[s:s + args.batch], return_tensors='pt', padding=True,
                      truncation=True, max_length=args.max_len).to('cuda')
            h = model(**enc).last_hidden_state
            idx = enc['attention_mask'].sum(1) - 1
            last = h[torch.arange(h.size(0), device=h.device), idx]
            out.extend(head(last.to(head.weight.dtype)).squeeze(-1).float().tolist())
        return out

    def fmt(q, sol):
        return tok.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': q},
             {'role': 'assistant', 'content': sol}],
            tokenize=False, add_generation_prompt=False)

    wins = margins = 0
    for i, r in enumerate(rows):
        sc, sr = score([fmt(r['question'], r['chosen']), fmt(r['question'], r['rejected'])])
        wins += (sc > sr)
        margins += (sc - sr)
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(rows)} — 현재 {wins/(i+1):.3f}', flush=True)

    n = len(rows)
    acc = wins / n
    se = math.sqrt(acc * (1 - acc) / n)
    print()
    print(f'holdout 쌍 정확도 : {acc:.3f}  (표준오차 {se:.3f}, 95% 구간 {acc-1.96*se:.3f}~{acc+1.96*se:.3f})')
    print(f'평균 점수차       : {margins/n:+.4f}')
    print()
    if acc < 0.55:
        print('→ 우연(0.5)과 구분되지 않음. 표현에도 정오 신호가 없다 = 선택 축 최종 폐쇄.')
    elif acc < 0.65:
        print('→ exp47 DPO(0.60)·exp18c와 같은 대역. 32개 중 고르는 실제 과제는 쌍 판별보다')
        print('   훨씬 어려우므로, 재순위화로 +1.5%p가 나올 가능성은 낮다.')
    else:
        print('→ 쌍 판별에는 유의미한 신호가 있다. Best-of-N 재순위화 평가로 진행할 가치가 있다.')

    os.makedirs('results', exist_ok=True)
    json.dump({'rm': args.rm, 'n_pairs': n, 'pair_acc': acc, 'stderr': se,
               'mean_margin': margins / n},
              open('results/eval_reward_holdout.json', 'w'), indent=2)
    print('저장: results/eval_reward_holdout.json')


if __name__ == '__main__':
    main()
