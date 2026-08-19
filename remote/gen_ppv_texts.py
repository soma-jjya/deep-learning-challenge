"""exp86 1단계 — 회수 가능 문항의 풀이 텍스트 재생성 (vLLM, 약 20분).

기존 덤프에는 답·logprob만 있고 텍스트가 없다. PPV 기하 진단에는 추론 내용이 필요하다.
선정: seed42 기준 회수 가능(SC오답 & gold 존재) & gold ≥2표 & 지배오답 ≥2표, 최대 50문항.
재실행이라 표본이 원본과 다를 수 있으므로 분석은 새 덤프 안에서 자기완결로 한다(사전등록).

사용: uv run python remote/gen_ppv_texts.py
"""
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

csv.field_size_limit(10 ** 7)
SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')
OUT = 'results/ppv_texts.jsonl'
MAXP = 50


def wvote(ss):
    v = [(s['ans'], s['logp']) for s in ss if s.get('ans') is not None]
    if not v:
        return None
    m = max(lp for _, lp in v)
    w = defaultdict(float)
    for a, lp in v:
        w[a] += math.exp((lp - m) * 2.0)
    return max(w, key=w.get)


def main():
    # ── 대상 선정 (기존 덤프 기준) ──
    picks = []
    for line in open('results/val_samples.jsonl', encoding='utf-8'):
        r = json.loads(line)
        ss = r['samples'][:32]
        g = r['gold']
        ans = [s['ans'] for s in ss if s.get('ans') is not None]
        if wvote(ss) == g or g not in ans:
            continue                       # 회수 가능 구간만
        c = Counter(ans)
        wrong = [(a, n) for a, n in c.items() if a != g]
        if c[g] >= 2 and wrong and max(n for _, n in wrong) >= 2:
            picks.append(r['id'])
    picks = picks[:MAXP]
    print('대상 %d문항' % len(picks))

    with open('deep-learning-challenge-2026/deep_chal_math_train.csv', encoding='utf-8') as f:
        qmap = {r['id']: (r['question'], int(r['answer'])) for r in csv.DictReader(f)}

    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)['id'] for l in open(OUT, encoding='utf-8')}
        print('이어하기: %d문항 완료' % len(done))
    todo = [p for p in picks if p not in done]
    if not todo:
        print('모두 완료')
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=32, temperature=0.7, top_p=0.8, max_tokens=1024,
                        seed=42, logprobs=0)

    with open(OUT, 'a', encoding='utf-8') as fo:
        for i in range(0, len(todo), 10):
            chunk = todo[i:i + 10]
            prompts = [tok.apply_chat_template(
                [{'role': 'system', 'content': SYSTEM_PROMPT},
                 {'role': 'user', 'content': qmap[p][0]}],
                tokenize=False, add_generation_prompt=True) for p in chunk]
            outs = llm.generate(prompts, sp)
            for pid, o in zip(chunk, outs):
                samples = [{'ans': extract_answer(c.text),
                            'logp': c.cumulative_logprob / max(1, len(c.token_ids)),
                            'len': len(c.token_ids),
                            'text': c.text} for c in o.outputs]
                fo.write(json.dumps({'id': pid, 'gold': qmap[pid][1],
                                     'samples': samples}, ensure_ascii=False) + chr(10))
            fo.flush()
            print('진행 %d/%d' % (min(i + 10, len(todo)), len(todo)), flush=True)
    print('저장: %s' % OUT)


if __name__ == '__main__':
    main()
