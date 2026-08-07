"""검증자 v2 학습 데이터 — '생각하고 판정' + 어려운 음성 강조 (H9 재도전).

v1 실패 진단 반영:
  ① 어려운 음성(4표본 중 다수답이 된 오답 = 다수결을 속인 풀이)을 전량 포함
  ② 즉답 Yes/No 대신, 베이스 모델이 풀이를 재검산한 근거+판정을 생성하고
     판정이 정답 라벨과 일치하는 근거만 채택 (판정 근거의 rejection sampling)
사용: nohup uv run python remote/gen_verifier_v2_data.py > gen_verifier_v2.log 2>&1 &
입력: data/verifier.jsonl (exp18a 산출물) / 출력: data/verifier_v2.jsonl
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

CONFIG = dict(
    src='data/verifier.jsonl', out='data/verifier_v2.jsonl',
    progress='data/verifier_v2_progress.json',
    max_easy_pos=4000, max_easy_neg=2000,   # 어려운 음성·양성은 전량, 쉬운 것만 다운샘플
    tries=2, temperature=0.7, max_tokens=600, chunk=400, seed=42,
)

JUDGE_GEN_PROMPT = (
    'You are a strict math solution grader. Re-derive the key steps of the proposed '
    'solution to check whether its final answer is correct. Be brief (a few lines). '
    'End with exactly one line: "Verdict: Yes" if the final answer is correct, '
    'or "Verdict: No" if it is not.')


def verdict_of(text):
    t = text.lower()
    i = t.rfind('verdict:')
    if i < 0:
        return None
    tail = t[i + 8:i + 20]
    if 'yes' in tail:
        return True
    if 'no' in tail:
        return False
    return None


def select_pairs():
    """문제별 4표본에서 다수답을 계산해 난이도 태그를 붙이고 선별."""
    by_pid = defaultdict(list)
    for line in open(CONFIG['src'], encoding='utf-8'):
        r = json.loads(line)
        by_pid[r['id']].append(r)

    hard_neg, easy_neg, hard_pos, easy_pos = [], [], [], []
    for pid, rows in by_pid.items():
        answers = [extract_answer(r['solution']) for r in rows]
        votes = [a for a in answers if a is not None]
        maj = Counter(votes).most_common(1)[0][0] if votes else None
        for r, a in zip(rows, answers):
            if r['label']:
                # 어려운 양성 = 정답인데 다수답이 아니었던 풀이 (표결에서 밀린 정답)
                (easy_pos if a == maj else hard_pos).append(r)
            else:
                # 어려운 음성 = 오답인데 다수답이 된 풀이 (다수결을 속인 풀이)
                (hard_neg if a is not None and a == maj else easy_neg).append(r)

    random.seed(CONFIG['seed'])
    random.shuffle(easy_pos)
    random.shuffle(easy_neg)
    sel = (hard_neg + hard_pos + easy_pos[:CONFIG['max_easy_pos']]
           + easy_neg[:CONFIG['max_easy_neg']])
    random.shuffle(sel)
    print(f'선별: 어려운음성 {len(hard_neg)} + 어려운양성 {len(hard_pos)} '
          f'+ 쉬운양성 {min(len(easy_pos), CONFIG["max_easy_pos"])} '
          f'+ 쉬운음성 {min(len(easy_neg), CONFIG["max_easy_neg"])} = {len(sel)}')
    return sel


def main():
    from vllm import LLM, SamplingParams

    pairs = select_pairs()
    done = set()
    if os.path.exists(CONFIG['progress']):
        done = set(json.load(open(CONFIG['progress']))['done'])
    todo = [(i, p) for i, p in enumerate(pairs) if i not in done]
    print(f'남은 {len(todo)}/{len(pairs)}쌍')

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.92, max_model_len=4096)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=CONFIG['tries'], temperature=CONFIG['temperature'],
                        top_p=0.9, max_tokens=CONFIG['max_tokens'], seed=11)

    kept = miss = 0
    for s in range(0, len(todo), CONFIG['chunk']):
        chunk = todo[s:s + CONFIG['chunk']]
        prompts = [tok.apply_chat_template(
            [{'role': 'system', 'content': JUDGE_GEN_PROMPT},
             {'role': 'user', 'content': 'Problem:' + chr(10) + p['question'] +
              chr(10) * 2 + 'Proposed solution:' + chr(10) + p['solution'][:3500]}],
            tokenize=False, add_generation_prompt=True) for _, p in chunk]
        outs = llm.generate(prompts, sp)
        with open(CONFIG['out'], 'a', encoding='utf-8') as f:
            for (idx, p), out in zip(chunk, outs):
                pick = None
                for c in out.outputs:
                    if verdict_of(c.text) == p['label']:
                        pick = c.text.strip()
                        break
                if pick:
                    kept += 1
                    f.write(json.dumps({'id': p['id'], 'question': p['question'],
                                        'solution': p['solution'], 'label': p['label'],
                                        'judge': pick}, ensure_ascii=False) + chr(10))
                else:
                    miss += 1
                done.add(idx)
        json.dump({'done': sorted(done)}, open(CONFIG['progress'], 'w'))
        print(f'진행 {len(done)}/{len(pairs)} | 채택 {kept} | 근거실패 {miss}')
    print(f'완료: {CONFIG["out"]} (채택 {kept}, 실패 {miss})')


if __name__ == '__main__':
    main()
