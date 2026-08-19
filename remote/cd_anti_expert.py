"""exp80a — 실패 체크포인트를 anti-expert로 쓰는 Contrastive Decoding.

가설(H35, 신규): exp76에서 계열 간 **오답 일치율 0.60~0.91**을 실측했다. 즉 실패한 체크포인트의
분포 안에는 base가 반복해서 빠지는 **공통 오류 모드**가 들어 있다. 앙상블에서는 그게 약점이었다
(같은 오답을 여러 번 세는 꼴). 그 분포를 **빼는 방향**으로 쓰면 교정 신호가 될 수 있다.

    score(t) = (1+λ)·log p_base(t) − λ·log p_amateur(t)       단, plausible 집합 안에서만
    plausible = { t : p_base(t) ≥ α · max_t p_base(t) }

⚠️ plausibility constraint는 선택이 아니라 필수다(arXiv:2210.15097). 없으면 amateur가 싫어한다는
   이유만으로 base가 거의 고려하지도 않은 토큰이 튀어나온다. **이 제약 없이 실패한 결과로
   CD 가설을 기각해서는 안 된다** — 사전등록에 못 박았다.

⚠️ λ=0 이면 정확히 base 스택으로 환원된다. 그래서 λ=0 실행이 **내장 대조군**이다.
   λ=0 결과가 base와 크게 다르면 구현이 틀린 것이지 가설이 틀린 것이 아니다.

amateur = `marginal` (exp68의 경계난이도 SFT, 자체 356 = base−12).
  remote/select_amateur.py 가 정량 선정: rescue율 0.017(최저) · contrast 9.9% · A=0.682.
  teacher는 rescue +10이라 제외했다 — 나쁜 모델이 아니라 다른 능력을 가진 specialist다.

base와 amateur는 **같은 베이스 모델 + LoRA**이므로 모델을 하나만 올리고
`disable_adapter()`로 두 분포를 얻는다 (메모리 1개분).

사용:
  python remote/cd_anti_expert.py --adapter <path> --ids-file <ids.json> --n 8 \
      --lam 0 0.5 1.0 --out results/cd_pilot.jsonl
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_extract import BS, extract_answer  # noqa: E402

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')


# ─────────────────────────────────────────────────────────────────────────────
# 순수 파이썬 참조 구현 — torch 없이 로컬에서 수식을 검증하기 위한 것.
# 실제 실행 경로(contrast_logits)와 같은 규칙을 따른다.
# ─────────────────────────────────────────────────────────────────────────────
def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def contrast_scores_ref(base_logits, ama_logits, lam, alpha):
    """(1+λ)·log p_base − λ·log p_amateur, plausible 밖은 -inf. 리스트 입출력."""
    pb = _softmax(base_logits)
    lb = [math.log(max(p, 1e-30)) for p in pb]
    la = [math.log(max(p, 1e-30)) for p in _softmax(ama_logits)]
    thr = alpha * max(pb)
    out = []
    for i in range(len(lb)):
        if pb[i] < thr:
            out.append(float('-inf'))
        else:
            out.append((1.0 + lam) * lb[i] - lam * la[i])
    return out


# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(tok, q):
    return tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--ids-file', required=True, help='대상 문항 id 목록 json')
    ap.add_argument('--train-csv', default='deep-learning-challenge-2026/deep_chal_math_train.csv')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--lam', type=float, nargs='+', default=[0.0, 0.5, 1.0])
    ap.add_argument('--alpha', type=float, default=0.1, help='plausibility 컷오프 (Li et al. 기본값)')
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=32, help='동시 생성 시퀀스 수')
    ap.add_argument('--out', default='results/cd_pilot.jsonl')
    ap.add_argument('--limit', type=int, default=None, help='앞 N문항만 (스모크)')
    ap.add_argument('--time-budget-min', type=float, default=120.0,
                    help='이 시간을 넘기면 남은 작업을 포기하고 정상 종료 (부분 결과는 보존)')
    args = ap.parse_args()

    import csv
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    csv.field_size_limit(10 ** 7)
    t0 = time.time()

    ids = json.load(open(args.ids_file, encoding='utf-8'))
    if args.limit:
        ids = ids[:args.limit]
    with open(args.train_csv, encoding='utf-8') as f:
        qmap = {r['id']: r['question'] for r in csv.DictReader(f)}
    todo_q = [(i, qmap[i]) for i in ids if i in qmap]
    print('대상 %d문항 · n=%d · λ=%s · α=%.2f' % (len(todo_q), args.n, args.lam, args.alpha))

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B-Instruct', torch_dtype=torch.bfloat16, device_map='cuda')
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print('모델 로드 완료 (%.0f초)' % (time.time() - t0))

    # 이어하기
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding='utf-8'):
            try:
                d = json.loads(line)
                done.add((d['id'], d['lam']))
            except Exception:
                pass
        print('이어서 진행: 완료 %d건' % len(done))

    eos = tok.eos_token_id
    extra_eos = set()
    for s in ('<|im_end|>', '<|endoftext|>'):
        t = tok.convert_tokens_to_ids(s)
        if isinstance(t, int) and t >= 0:
            extra_eos.add(t)
    extra_eos.add(eos)

    @torch.no_grad()
    def generate(prompts, lam):
        """prompts: list[str] → list[(text, mean_logprob, trunc, ntok)]"""
        enc = tok(prompts, return_tensors='pt', padding=True).to('cuda')
        ids_ = enc.input_ids
        attn = enc.attention_mask
        B = ids_.shape[0]
        pkv_e = pkv_a = None
        cur = ids_
        alive = torch.ones(B, dtype=torch.bool, device='cuda')
        out_toks = [[] for _ in range(B)]
        out_lp = [0.0] * B
        for step in range(args.max_tokens):
            with model.disable_adapter():
                oe = model(input_ids=cur, attention_mask=attn,
                           past_key_values=pkv_e, use_cache=True)
            oa = model(input_ids=cur, attention_mask=attn,
                       past_key_values=pkv_a, use_cache=True)
            pkv_e, pkv_a = oe.past_key_values, oa.past_key_values
            lb = torch.log_softmax(oe.logits[:, -1, :].float(), dim=-1)
            la = torch.log_softmax(oa.logits[:, -1, :].float(), dim=-1)
            # plausibility: p_base >= alpha * max p_base  ⇔  lb >= log(alpha) + max(lb)
            keep = lb >= (math.log(args.alpha) + lb.max(dim=-1, keepdim=True).values)
            score = (1.0 + lam) * lb - lam * la
            score = score.masked_fill(~keep, float('-inf'))
            # 우리 스택과 같은 샘플링 (temperature, top_p)
            probs = torch.softmax(score / args.temp, dim=-1)
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cdf = sp.cumsum(dim=-1)
            cut = (cdf - sp) > args.top_p
            sp = sp.masked_fill(cut, 0.0)
            sp = sp / sp.sum(dim=-1, keepdim=True)
            pick = torch.multinomial(sp, 1)
            nxt = si.gather(-1, pick).squeeze(-1)
            # 채택 확률은 **base 분포 기준**으로 기록한다 (기존 스택과 같은 자를 쓰기 위해)
            step_lp = lb.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
            for b in range(B):
                if alive[b]:
                    out_toks[b].append(int(nxt[b]))
                    out_lp[b] += float(step_lp[b])
                    if int(nxt[b]) in extra_eos:
                        alive[b] = False
            if not bool(alive.any()):
                break
            cur = nxt.unsqueeze(-1)
            attn = torch.cat([attn, alive.long().unsqueeze(-1)], dim=-1)
        res = []
        for b in range(B):
            tks = out_toks[b]
            trunc = len(tks) >= args.max_tokens and (not tks or tks[-1] not in extra_eos)
            txt = tok.decode(tks, skip_special_tokens=True)
            res.append((txt, out_lp[b] / max(1, len(tks)), trunc, len(tks)))
        return res

    os.makedirs('results', exist_ok=True)
    # ⚠️ 문항 하나씩 돌리면 배치가 n(=8)에 묶여 GPU가 논다 — 스모크에서 3문항 3.1분이었다.
    #    **여러 문항의 표본을 한 배치로 묶는다**(좌측 패딩이라 프롬프트 길이가 달라도 된다).
    #    같은 λ 안에서만 묶는다 — λ가 다르면 점수 계산이 달라지기 때문이다.
    work = []
    for lam in args.lam:
        for pid, q in todo_q:
            if (pid, lam) in done:
                continue
            pr = build_prompt(tok, q)
            work.extend([(pid, lam, pr)] * args.n)
    # λ별로 인접하게 정렬(안정 정렬이라 문항 순서는 유지된다)
    work.sort(key=lambda x: args.lam.index(x[1]))
    total_seq = len(work)
    print('생성할 시퀀스 %d개 (배치 %d) — 이미 완료된 것은 제외됨' % (total_seq, args.batch))

    acc = {}
    n_done_seq = 0
    with open(args.out, 'a', encoding='utf-8') as fo:
        for s in range(0, len(work), args.batch):
            if (time.time() - t0) / 60.0 > args.time_budget_min:
                print('⏹ 시간 예산 %.0f분 초과 — 남은 작업을 중단한다 (부분 결과 보존)'
                      % args.time_budget_min, flush=True)
                break
            chunk = work[s:s + args.batch]
            # 한 배치 안에 λ가 섞이면 안 된다
            lam = chunk[0][1]
            chunk = [c for c in chunk if c[1] == lam]
            for (pid, _, _), (txt, lp, tr, nt) in zip(chunk, generate([c[2] for c in chunk], lam)):
                acc.setdefault((pid, lam), []).append(
                    {'ans': extract_answer(txt), 'logp': lp, 'trunc': tr, 'len': nt})
            n_done_seq += len(chunk)
            # n개가 모인 문항은 즉시 기록 (중간에 죽어도 보존)
            for key in [k for k, v in acc.items() if len(v) >= args.n]:
                fo.write(json.dumps({'id': key[0], 'lam': key[1], 'samples': acc.pop(key)},
                                    ensure_ascii=False) + chr(10))
            fo.flush()
            el = (time.time() - t0) / 60.0
            print('  시퀀스 %d/%d · 경과 %.1f분 · 예상 총 %.1f분'
                  % (n_done_seq, total_seq, el, el / max(1, n_done_seq) * total_seq), flush=True)
        # 남은 조각도 버리지 않는다
        for key, v in acc.items():
            fo.write(json.dumps({'id': key[0], 'lam': key[1], 'samples': v,
                                 'partial': True}, ensure_ascii=False) + chr(10))
    print('저장: %s (시퀀스 %d/%d, 경과 %.1f분)'
          % (args.out, n_done_seq, total_seq, (time.time() - t0) / 60.0))


if __name__ == '__main__':
    main()
