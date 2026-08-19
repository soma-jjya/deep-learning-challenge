"""리더보드 표본 대량 덤프 — 한 번 생성해두고 제출 파일을 무한히 찍어내기 위한 재료.

지금까지는 제출 후보 1개당 GPU 1~2시간을 썼다(그래서 하루 1~2개가 한계).
표본을 한 번(n=96) 만들어 저장하면, 이후 어떤 n·어떤 집계 규칙의 제출 파일도
GPU 없이 수 초 만에 만들 수 있다 → 하루 최대 건수 제출이 가능해진다.

사용: nohup uv run python remote/dump_lb_samples.py --n 96 > dump_lb.log 2>&1 &
출력: results/lb_samples.jsonl  (용량 커서 커밋 금지 — 서버에만 보관)
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=96)
    ap.add_argument('--seed', type=int, default=42, help='vLLM 샘플링 시드 (분산 측정용 다중 표본 세트 생성 시 변경)')
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--chunk', type=int, default=100, help='이 단위로 생성·저장 (OOM 방지·재개 지원)')
    ap.add_argument('--lb-csv', default='deep-learning-challenge-2026/deep_chal_math_leaderboard_filtered.csv')
    ap.add_argument('--out', default='results/lb_samples.jsonl')
    # 8/31 당일용: 처음 보는 test 파일에 2시간 30분을 걸기 전에, 몇 문항으로 경로 전체를
    # 먼저 통과시켜 본다. 컬럼명·인코딩·문항 수 같은 것이 어긋나면 여기서 즉시 드러난다.
    ap.add_argument('--limit', type=int, default=None,
                    help='앞에서 N문항만 (스모크 테스트용 — 실전에서는 지정하지 말 것)')
    # 로컬 3시드가 노이즈여도 리더보드로 판정을 받아야 하는 후보가 있다. 우리 정책이
    # "로컬은 오답의 23%가 라벨 의심인 흐린 자, 리더보드는 운영진 검수를 거친 깨끗한 자"
    # 이므로 더 좋은 측정기를 아낄 이유가 없다(exp41은 로컬과 LB가 정반대였다).
    ap.add_argument('--adapter', default=None, help='LoRA 어댑터 경로 (없으면 베이스)')
    # ⚠️ 2026-08-19 추가 (8/31 신뢰성 감사). 아래 재개 로직은 done 목록만 보고 이어붙이므로,
    # 다른 --n/--seed/--temp/--top-p/--max-tokens/--adapter 로 재개해도 **경고 없이 섞인다.**
    # 당일 2시간 30분짜리 작업이 중단됐을 때 명령을 다시 타이핑하다 한 글자 틀리면
    # 앞뒤 파라미터가 다른 덤프가 만들어지고, 그 사실이 어디에도 남지 않는다.
    ap.add_argument('--force-resume', action='store_true',
                    help='파라미터가 이전 실행과 달라도 이어서 진행 (권장하지 않음)')
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lb = pd.read_csv(args.lb_csv)
    lb.columns = lb.columns.str.strip()
    # 8/31 test 파일은 처음 보는 파일이다. 컬럼명이 조금만 달라도(대문자·problem 등)
    # 아래에서 KeyError로 죽는데, 그때 원인을 찾느라 시간을 태우지 않도록 여기서 맞춰준다.
    ren = {}
    for want, alts in (('id', ('id', 'ID', 'Id', 'problem_id', 'index')),
                       ('question', ('question', 'Question', 'problem', 'Problem', 'text'))):
        if want in lb.columns:
            continue
        hit = next((c for c in lb.columns if c in alts), None)
        if hit is None:
            hit = next((c for c in lb.columns if c.lower() == want), None)
        if hit is None:
            raise SystemExit('⛔ test 파일에 %r 컬럼이 없다. 실제 컬럼: %s'
                             % (want, list(lb.columns)))
        ren[hit] = want
    if ren:
        print('⚠️ 컬럼명 자동 매핑: %s' % ren)
        lb = lb.rename(columns=ren)
    if lb['id'].duplicated().any():
        raise SystemExit('⛔ test 파일에 중복 id가 있다 — 진행하면 덤프가 꼬인다')
    if args.limit:
        lb = lb.iloc[:args.limit].reset_index(drop=True)
        print(f'⚠️ 스모크 테스트 모드 — 앞 {len(lb)}문항만 처리한다 (실전 제출용 아님)')
    print(f'리더보드 {len(lb)}문항 × {args.n}샘플 덤프 시작 (seed={args.seed})')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096,
              enable_lora=args.adapter is not None, max_lora_rank=64)
    lora = LoRARequest('adapter', 1, args.adapter) if args.adapter else None
    if lora:
        print(f'어댑터 적용: {args.adapter}')
    tok = llm.get_tokenizer()

    # ── 청크 처리 + 중간 저장 (2026-08-10) ──
    # 831문항 × 96샘플을 한 번에 생성하면 결과가 통째로 RAM에 쌓여 OOM으로 죽는다
    # (실제로 exp29가 이 방식으로 사망). 청크마다 디스크에 쓰고 메모리를 비우면
    # 사용량이 1/N로 줄고, 중간에 죽어도 done 목록을 보고 이어서 재개된다.
    os.makedirs('results', exist_ok=True)
    prog = args.out + '.progress'
    # 재개 시 파라미터가 같은지 대조하기 위한 지문 (--limit은 스모크용이라 제외하지 않는다)
    fp = {'n': args.n, 'seed': args.seed, 'temp': args.temp, 'top_p': args.top_p,
          'max_tokens': args.max_tokens, 'adapter': args.adapter,
          'lb_csv': os.path.basename(args.lb_csv), 'limit': args.limit}
    done = set()
    if os.path.exists(prog) and os.path.exists(args.out):
        st = json.load(open(prog))
        old = st.get('params')
        if old is None:
            print('⚠️ 이전 진행 파일에 파라미터 지문이 없다(구 버전에서 생성됨) — 대조 생략')
        elif old != fp:
            diff = {k: (old.get(k), fp[k]) for k in fp if old.get(k) != fp[k]}
            msg = ('⛔ 이어서 진행하려는 설정이 이전 실행과 다르다: %s' % diff + chr(10) +
                   '   그대로 진행하면 앞부분과 뒷부분의 설정이 다른 덤프가 만들어진다.' + chr(10) +
                   '   같은 설정으로 다시 실행하거나, 새 --out 경로를 쓰거나, '
                   '정말 의도한 것이면 --force-resume 을 붙일 것.' + chr(10) +
                   '   ※ 시간이 부족해 --n 을 32→16으로 낮추는 경우라면 --force-resume 이 맞다. '
                   '앞부분에 표본이 더 많을 뿐이고, 제출 시 --n 16 으로 자르면 전 문항이 같아진다.')
            if not args.force_resume:
                raise SystemExit(msg)
            print(msg + chr(10) + '   (--force-resume 지정됨 — 그대로 진행)')
        done = set(st['done'])
        # 청크 도중에 죽으면 jsonl에는 쓰였는데 done에는 없는 문항이 생긴다.
        # 그대로 두면 같은 id가 두 번 기록되므로, 실제 파일을 읽어 done을 보정한다.
        on_disk = set()
        with open(args.out, encoding='utf-8') as f:
            for line in f:
                try:
                    on_disk.add(json.loads(line)['id'])
                except Exception:
                    pass
        extra = on_disk - done
        if extra:
            print(f'ℹ️ 진행 파일에는 없지만 덤프에 이미 있는 문항 {len(extra)}개 — 재생성하지 않는다')
            done |= extra
        print(f'이어서 진행: {len(done)}문항 완료됨')

    todo = [(i, q) for i, q in enumerate(lb['question']) if lb['id'][i] not in done]
    sp = SamplingParams(n=args.n, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed, logprobs=0)

    for s in range(0, len(todo), args.chunk):
        part = todo[s:s + args.chunk]
        prompts = [tok.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': q}],
            tokenize=False, add_generation_prompt=True) for _, q in part]
        outs = llm.generate(prompts, sp, lora_request=lora)
        with open(args.out, 'a', encoding='utf-8') as f:
            for (i, _), o in zip(part, outs):
                samples = []
                for c in o.outputs:
                    ntok = max(1, len(c.token_ids))
                    samples.append({'ans': extract_answer(c.text),
                                    'logp': c.cumulative_logprob / ntok,
                                    'trunc': c.finish_reason == 'length',
                                    'len': ntok})
                f.write(json.dumps({'id': lb['id'][i], 'samples': samples},
                                   ensure_ascii=False) + chr(10))
                done.add(lb['id'][i])
        json.dump({'done': sorted(done), 'params': fp}, open(prog, 'w'))
        del outs
        print(f'진행 {len(done)}/{len(lb)}문항 저장됨')
    print(f'저장: {args.out} ({len(lb)}문항 × {args.n})')
    print('다음: remote/make_submission_from_dump.py 로 원하는 만큼 제출 파일 생성 (GPU 불필요)')


if __name__ == '__main__':
    main()
