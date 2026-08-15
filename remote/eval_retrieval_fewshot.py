"""H26 — 검색 기반 퓨샷: 문제마다 **비슷한 풀린 문제**를 찾아 예시로 붙인다.

exp20(퓨샷)과 무엇이 다른가: exp20은 기하·대수·정수론·응용에서 고른 예시 **4개를 모든
문제에 똑같이** 붙였고 SC가 74.7%→73.1%로 떨어졌다. 무관한 예시는 문체만 고정시키고
방해가 된다. 여기서는 **문제별로 다른 예시**를 검색해서 붙인다 — 조합론 문제에는 조합론
풀이를, 정수론 문제에는 정수론 풀이를 보여준다.

왜 지금 유망한가:
  - 분야별 진단(analyze_by_category)에서 조합론 30.0%, 정수론·기하가 특히 약했다.
    유형별 풀이 패턴을 못 잡는 것이 원인이라면, 그 유형의 풀린 예시가 직접적인 처방이다.
  - 재료가 이미 있다: `data/sft_short.jsonl`(문제당 최단 정답 풀이). 36k 규모.
  - 학습이 아니라 **추론 시점 조건화**라 학습 7전이 부딪힌 벽(기존 조율 파괴)을 우회한다.

규칙 적합성: 검색은 TF-IDF(순수 통계, 모델 아님)이고 예시는 우리 train 데이터다.
추론 시 외부 모델·인터넷·코드 실행을 쓰지 않는다(prd.md:35).

⚠️ sklearn을 설치하지 않고 TF-IDF를 직접 구현한다 — 과거 mergekit 설치가 transformers를
다운그레이드해 vLLM(=최종 파이프라인)을 위협한 적이 있다. 의존성을 늘리지 않는다.

사용: uv run python remote/eval_retrieval_fewshot.py --k 3 --n 32 --seed 42
"""
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import pandas as pd

BS = chr(92)
SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')

TOKEN_PAT = re.compile(r'[a-zA-Z]{2,}')
STOP = set('the a an of to in is are and or for that this with be we can if it as by on '
           'find such what how many number let where when then all each are'.split())


def tokenize(text):
    return [t for t in TOKEN_PAT.findall(text.lower()) if t not in STOP]


class TfidfIndex:
    """역색인 + 코사인 유사도. 외부 의존성 없음."""

    def __init__(self, docs):
        self.docs = docs                       # [(id, question, solution)]
        self.df = Counter()
        self.tf = []
        for _, q, _ in docs:
            c = Counter(tokenize(q))
            self.tf.append(c)
            self.df.update(c.keys())
        self.N = len(docs)
        self.idf = {t: math.log((self.N + 1) / (d + 1)) + 1.0 for t, d in self.df.items()}
        # 문서 벡터 정규화 + 역색인
        self.norm = []
        self.inv = defaultdict(list)
        for i, c in enumerate(self.tf):
            s = 0.0
            for t, f in c.items():
                w = (1 + math.log(f)) * self.idf.get(t, 0.0)
                s += w * w
                self.inv[t].append((i, w))
            self.norm.append(math.sqrt(s) or 1.0)

    def search(self, query, k=3, exclude=()):
        c = Counter(tokenize(query))
        scores = defaultdict(float)
        qn = 0.0
        for t, f in c.items():
            w = (1 + math.log(f)) * self.idf.get(t, 0.0)
            qn += w * w
            # 너무 흔한 단어는 건너뛴다 — 변별력이 없는데 비용만 크다
            post = self.inv.get(t)
            if not post or len(post) > self.N * 0.25:
                continue
            for i, dw in post:
                scores[i] += w * dw
        qn = math.sqrt(qn) or 1.0
        ranked = sorted(((s / (self.norm[i] * qn), i) for i, s in scores.items()),
                        reverse=True)
        out = []
        for sc, i in ranked:
            if self.docs[i][0] in exclude:
                continue
            out.append((sc, self.docs[i]))
            if len(out) >= k:
                break
        return out


def load_pool(path, max_sol_chars):
    """{"messages":[system,user,assistant]} 형식에서 (id, 문제, 풀이)를 뽑는다."""
    pool = []
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        msgs = r.get('messages') or []
        q = next((m['content'] for m in msgs if m['role'] == 'user'), None)
        a = next((m['content'] for m in msgs if m['role'] == 'assistant'), None)
        if not q or not a or len(a) > max_sol_chars:
            continue
        pool.append((r.get('id', ''), q, a))
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='data/sft_short.jsonl')
    ap.add_argument('--k', type=int, default=3, help='붙일 예시 수')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--max-model-len', type=int, default=8192,
                    help='예시가 들어가므로 기본 4096으로는 부족하다 (Qwen2.5는 32k 지원)')
    ap.add_argument('--max-sol-chars', type=int, default=2500,
                    help='이보다 긴 예시 풀이는 문맥만 먹으므로 제외')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    ap.add_argument('--dry-run', action='store_true', help='검색 품질만 확인하고 종료(GPU 불필요)')
    args = ap.parse_args()

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        val = val[~val['id'].isin(set(pd.read_csv(bad)['id']))].reset_index(drop=True)
    val_ids = set(val['id'])

    pool = load_pool(args.pool, args.max_sol_chars)
    # 검증셋 문항이 예시로 새어 들어가면 정답을 그대로 보여주는 셈이다 — 반드시 제외
    pool = [d for d in pool if d[0] not in val_ids]
    print(f'검증 {len(val)}문항 / 검색 풀 {len(pool)}개 (검증셋 제외 완료)')

    idx = TfidfIndex(pool)
    print('색인 완료. 검색 중...')
    retrieved = [idx.search(q, k=args.k, exclude=val_ids) for q in val['question']]
    sims = [r[0][0] for r in retrieved if r]
    print(f'평균 1위 유사도 {sum(sims)/max(1,len(sims)):.3f}, '
          f'예시를 못 찾은 문항 {sum(1 for r in retrieved if not r)}개')

    if args.dry_run:
        print()
        for i in range(3):
            print('=' * 70)
            print('[대상]', val['question'][i][:180].replace(chr(10), ' '))
            for sc, (pid, q, a) in retrieved[i]:
                print(f'  ({sc:.3f}) {pid}: {q[:150]}'.replace(chr(10), ' '))
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=args.max_model_len)
    tok = llm.get_tokenizer()

    prompts = []
    for q, rs in zip(val['question'], retrieved):
        msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        for _, (_, eq, ea) in rs:
            msgs.append({'role': 'user', 'content': eq})
            msgs.append({'role': 'assistant', 'content': ea})
        msgs.append({'role': 'user', 'content': q})
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))
    ntok = [len(tok(p)['input_ids']) for p in prompts]
    print(f'프롬프트 토큰: 중앙값 {sorted(ntok)[len(ntok)//2]}, 최대 {max(ntok)} '
          f'(max_model_len={args.max_model_len})')

    outs = llm.generate(prompts, SamplingParams(
        n=args.n, temperature=args.temp, top_p=args.top_p,
        max_tokens=args.max_tokens, seed=args.seed, logprobs=0))

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from answer_extract import extract_answer

    out = args.out or f'results/val_samples_rfs{args.k}_s{args.seed}.jsonl'
    os.makedirs('results', exist_ok=True)
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
    print(f'저장: {out}')
    print('다음: remote/analyze_selection_gap.py --samples ' + out + ' --n 32')


if __name__ == '__main__':
    main()
