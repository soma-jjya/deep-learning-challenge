"""exp78a — MTI (Minimal Test-Time Intervention, arXiv:2510.13940, ACL 2026).

    log P̂(x_t) = (1−ω)·log P(x_t | c̄, x_<t) + ω·log P(x_t | c, x_<t)      단 H(p) > τ 인 위치만
    ω = 1.5 (논문 기본값) · c̄ = 현재 접두 뒤에 "OUTPUT ERROR"를 붙인 무조건 분기

exp80a의 대조 디코딩과 **수식 형태가 같다**(λ = ω−1). 다른 것은 둘뿐이다.
  ① 두 번째 분기가 실패 LoRA가 아니라 **"OUTPUT ERROR" 주입 프롬프트**
  ② **엔트로피 > τ 인 위치에서만** 개입한다 (논문: τ=1.5면 토큰의 4.2%, τ=0.5면 31.2%)

⚠️ vLLM을 건드리지 않는다. 논문 공식 구현은 vLLM 내부를 수정하지만, 여기서는 이미 검증된
   exp80a의 HF 디코딩 루프를 재사용한다 — mergekit 사고(의존성 다운그레이드)의 재발 위험을
   구조적으로 없앤다.

⚠️ 무조건 분기는 논문의 방식대로 **KV 캐시를 재사용**한다: 개입 지점에서 현재 캐시에
   "OUTPUT ERROR" 토큰만 덧붙여 forward하고, 다음 토큰 분포를 얻은 뒤 캐시를 원래 길이로
   되돌린다(crop). 그래서 개입하지 않는 위치에서는 추가 비용이 0이다.

⚠️ 하이퍼파라미터를 검증셋에서 탐색하지 않는다. **논문이 쓴 값을 그대로 쓴다**:
   ω=1.5, τ ∈ {0.5, 1.5}. 우리 모델에서 실제 개입 비율이 몇 %인지는 결과로 보고한다.
   (우리 모델은 short-CoT non-thinking instruct이고 논문은 long-CoT 추론 모델이다 —
    exp61·exp62에서 이 계열 차이로 외삽이 두 번 깨졌으므로 그 가능성을 열어둔다.)

사용:
  python remote/mti_decode.py --ids-file experiments/exp80a_ids60.json --n 8 \
      --omega 1.5 --tau 1.5 0.5 --out results/mti_pilot.jsonl
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_extract import BS, extract_answer  # noqa: E402
from cd_anti_expert import SYSTEM_PROMPT, build_prompt  # noqa: E402

NEG_PHRASE = ' OUTPUT ERROR'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids-file', required=True)
    ap.add_argument('--train-csv', default='deep-learning-challenge-2026/deep_chal_math_train.csv')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--omega', type=float, default=1.5, help='논문 기본값 1.5')
    ap.add_argument('--tau', type=float, nargs='+', default=[1.5, 0.5],
                    help='엔트로피 임계 (논문이 보고한 두 값)')
    ap.add_argument('--control', action='store_true', help='개입 없는 대조군도 함께 (tau=inf)')
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--out', default='results/mti_pilot.jsonl')
    ap.add_argument('--time-budget-min', type=float, default=60.0)
    args = ap.parse_args()

    import csv
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    csv.field_size_limit(10 ** 7)
    t0 = time.time()
    taus = list(args.tau) + ([float('inf')] if args.control else [])

    ids = json.load(open(args.ids_file, encoding='utf-8'))
    with open(args.train_csv, encoding='utf-8') as f:
        qmap = {r['id']: r['question'] for r in csv.DictReader(f)}
    todo_q = [(i, qmap[i]) for i in ids if i in qmap]
    print('대상 %d문항 · n=%d · ω=%.2f · τ=%s' % (len(todo_q), args.n, args.omega, taus))

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B-Instruct', torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    neg_ids = tok(NEG_PHRASE, add_special_tokens=False).input_ids
    print('모델 로드 완료 (%.0f초) · 부정 문구 토큰 %d개' % (time.time() - t0, len(neg_ids)))

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding='utf-8'):
            try:
                d = json.loads(line)
                done.add((d['id'], d['tau']))
            except Exception:
                pass
        print('이어서 진행: 완료 %d건' % len(done))

    extra_eos = {tok.eos_token_id}
    for s in ('<|im_end|>', '<|endoftext|>'):
        t = tok.convert_tokens_to_ids(s)
        if isinstance(t, int) and t >= 0:
            extra_eos.add(t)

    def cache_len(pkv):
        try:
            return pkv.get_seq_length()
        except Exception:
            return pkv[0][0].shape[-2]

    @torch.no_grad()
    def generate(prompts, tau):
        enc = tok(prompts, return_tensors='pt', padding=True).to('cuda')
        attn = enc.attention_mask
        B = enc.input_ids.shape[0]
        pkv = None
        cur = enc.input_ids
        alive = torch.ones(B, dtype=torch.bool, device='cuda')
        toks = [[] for _ in range(B)]
        lps = [0.0] * B
        n_step = n_interv = 0
        neg_t = torch.tensor([neg_ids] * B, device='cuda')
        for _ in range(args.max_tokens):
            out = model(input_ids=cur, attention_mask=attn, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            lc = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
            ent = -(lc.exp() * lc).sum(dim=-1)          # Shannon entropy, nats
            need = (ent > tau) & alive
            n_step += int(alive.sum())
            score = lc
            if bool(need.any()) and math.isfinite(tau):
                n_interv += int(need.sum())
                keep_len = cache_len(pkv)
                # 무조건 분기: 현재 캐시 뒤에 "OUTPUT ERROR"만 덧붙여 forward
                nattn = torch.cat([attn, torch.ones(B, len(neg_ids), dtype=attn.dtype,
                                                    device='cuda')], dim=-1)
                uo = model(input_ids=neg_t, attention_mask=nattn,
                           past_key_values=pkv, use_cache=True)
                lu = torch.log_softmax(uo.logits[:, -1, :].float(), dim=-1)
                try:
                    pkv.crop(keep_len)               # 캐시를 원래 길이로 되돌린다
                except Exception:
                    raise SystemExit('⛔ KV 캐시 crop을 지원하지 않는 transformers 버전이다')
                cfg = (1.0 - args.omega) * lu + args.omega * lc
                score = torch.where(need.unsqueeze(-1), cfg, lc)
            probs = torch.softmax(score / args.temp, dim=-1)
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cdf = sp.cumsum(dim=-1)
            sp = sp.masked_fill((cdf - sp) > args.top_p, 0.0)
            sp = sp / sp.sum(dim=-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1)).squeeze(-1)
            step_lp = lc.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)   # 자는 base 분포로 통일
            for b in range(B):
                if alive[b]:
                    toks[b].append(int(nxt[b]))
                    lps[b] += float(step_lp[b])
                    if int(nxt[b]) in extra_eos:
                        alive[b] = False
            if not bool(alive.any()):
                break
            cur = nxt.unsqueeze(-1)
            attn = torch.cat([attn, alive.long().unsqueeze(-1)], dim=-1)
        res = []
        for b in range(B):
            tk = toks[b]
            tr = len(tk) >= args.max_tokens and (not tk or tk[-1] not in extra_eos)
            res.append((tok.decode(tk, skip_special_tokens=True),
                        lps[b] / max(1, len(tk)), tr, len(tk)))
        return res, (n_interv / max(1, n_step))

    work = []
    for tau in taus:
        for pid, q in todo_q:
            if (pid, tau) in done:
                continue
            work.extend([(pid, tau, build_prompt(tok, q))] * args.n)
    work.sort(key=lambda x: taus.index(x[1]))
    print('생성할 시퀀스 %d개 (배치 %d)' % (len(work), args.batch))

    os.makedirs('results', exist_ok=True)
    acc, seen, frac = {}, 0, {}
    with open(args.out, 'a', encoding='utf-8') as fo:
        for s in range(0, len(work), args.batch):
            if (time.time() - t0) / 60.0 > args.time_budget_min:
                print('⏹ 시간 예산 초과 — 중단 (부분 결과 보존)', flush=True)
                break
            chunk = work[s:s + args.batch]
            tau = chunk[0][1]
            chunk = [c for c in chunk if c[1] == tau]
            outs, fr = generate([c[2] for c in chunk], tau)
            frac.setdefault(tau, []).append(fr)
            for (pid, _, _), (txt, lp, tr, nt) in zip(chunk, outs):
                acc.setdefault((pid, tau), []).append(
                    {'ans': extract_answer(txt), 'logp': lp, 'trunc': tr, 'len': nt})
            seen += len(chunk)
            for k in [k for k, v in acc.items() if len(v) >= args.n]:
                fo.write(json.dumps({'id': k[0], 'tau': k[1], 'omega': args.omega,
                                     'interv_frac': sum(frac[k[1]]) / len(frac[k[1]]),
                                     'samples': acc.pop(k)}, ensure_ascii=False) + chr(10))
            fo.flush()
            el = (time.time() - t0) / 60.0
            print('  시퀀스 %d/%d · τ=%.2f 개입비율 %.1f%% · 경과 %.1f분 · 예상 %.1f분'
                  % (seen, len(work), tau, 100 * fr, el, el / max(1, seen) * len(work)),
                  flush=True)
    print('저장: %s (시퀀스 %d/%d, %.1f분)' % (args.out, seen, len(work), (time.time() - t0) / 60))
    for tau, v in frac.items():
        print('  τ=%.2f 평균 개입 비율 %.1f%% (논문: τ=1.5→4.2%%, τ=0.5→31.2%%)'
              % (tau, 100 * sum(v) / len(v)))


if __name__ == '__main__':
    main()
