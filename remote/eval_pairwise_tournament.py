"""쌍대 비교 토너먼트 (H9 대안) — 검증자(독립 채점)가 실패했으니 상대 비교를 시도.

results/val_samples.jsonl(exp31a, n=32)에서 상위 2개 후보 답의 표 차이가 margin 이하인
"접전" 문제만 골라, 그 문제만 새로 n개 풀이를 생성해 각 후보 답의 대표 풀이(최고 확신도)를
확보한다. 두 풀이를 나란히 제시하고 베이스 모델에게 "A/B 중 어느 쪽이 옳은가"를 5회 물어
다수결로 승자를 정하고, 그 답으로 해당 문제의 최종 답을 교체한다. 비접전 문제는 그대로
확신도 가중 다수결(scale=2.0, 기존 75.8% 스택과 동일 규칙) 답을 사용한다.

사용: uv run python remote/eval_pairwise_tournament.py --margin 3 --gen-n 8 --votes 5
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')
JUDGE_SYSTEM = (
    'You are an expert math competition judge. You will see a problem and two candidate '
    'solutions, A and B, that reach different final answers. Decide which solution is '
    'correct by checking the reasoning step by step. '
    'Reply with exactly one character: A or B.')


def softmax_weights(logps, scale):
    m = max(logps)
    return [math.exp((lp - m) * scale) for lp in logps]


def weighted_vote(samples, scale=2.0):
    v = [(s['ans'], s['logp']) for s in samples if s['ans'] is not None]
    if not v:
        return 0
    w = defaultdict(float)
    for (a, lp), ww in zip(v, softmax_weights([x[1] for x in v], scale)):
        w[a] += ww
    return max(w, key=w.get)


def top2_by_votes(samples, n):
    ans = [s['ans'] for s in samples[:n] if s['ans'] is not None]
    mc = Counter(ans).most_common(2)
    if len(mc) < 2:
        return None
    (a1, v1), (a2, v2) = mc
    return a1, v1, a2, v2


def load_val():
    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123).reset_index(drop=True)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        bad_ids = set(pd.read_csv(bad)['id'])
        val = val[~val['id'].isin(bad_ids)].reset_index(drop=True)
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='results/val_samples.jsonl')
    ap.add_argument('--n', type=int, default=32, help='표 계산에 쓸 표본 수 (기준 스택과 동일)')
    ap.add_argument('--margin', type=int, default=3, help='접전 판정 기준 (top1표-top2표 <= margin)')
    ap.add_argument('--gen-n', type=int, default=8, help='접전 문제당 대표 풀이 확보용 재생성 수')
    ap.add_argument('--votes', type=int, default=5, help='쌍대 비교 질의 횟수')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples, encoding='utf-8')]
    val = load_val()
    assert len(val) == len(rows), f'{len(val)} vs {len(rows)}'
    qmap = {row_id: q for row_id, q in zip(val['id'], val['question'])}

    baseline = {}
    contested = []  # (id, a1, a2)
    for r in rows:
        baseline[r['id']] = weighted_vote(r['samples'][:args.n])
        t = top2_by_votes(r['samples'], args.n)
        if t is None:
            continue
        a1, v1, a2, v2 = t
        if v1 - v2 <= args.margin:
            contested.append((r['id'], a1, a2))

    print(f'전체 {len(rows)}문항, 기준(가중 n={args.n}) 사용, 접전(margin<={args.margin}) {len(contested)}문항')

    from vllm import LLM, SamplingParams

    llm = LLM(model='Qwen/Qwen2.5-3B-Instruct', dtype='bfloat16',
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()

    # 1단계: 접전 문제만 재생성해 후보 답별 대표 풀이(최고 확신도) 확보
    gen_prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': qmap[cid]}],
        tokenize=False, add_generation_prompt=True) for cid, _, _ in contested]
    gen_outs = llm.generate(gen_prompts, SamplingParams(
        n=args.gen_n, temperature=0.7, top_p=0.8, max_tokens=2048, seed=42))

    reps = {}  # id -> {answer: (text, logp)}
    for (cid, a1, a2), o in zip(contested, gen_outs):
        best = {}
        for c in o.outputs:
            a = extract_answer(c.text)
            if a is None:
                continue
            ntok = max(1, len(c.token_ids))
            lp = c.cumulative_logprob / ntok
            if a not in best or lp > best[a][1]:
                best[a] = (c.text.strip(), lp)
        reps[cid] = best

    # 2단계: 대표 풀이를 둘 다 확보한 문제만 토너먼트 진행
    tourney = [(cid, a1, a2) for cid, a1, a2 in contested
               if a1 in reps[cid] and a2 in reps[cid]]
    skipped = len(contested) - len(tourney)
    print(f'대표 풀이 확보(양쪽 다) {len(tourney)}문항, 확보 실패(스킵, 기준값 유지) {skipped}문항')

    judge_prompts = []
    for cid, a1, a2 in tourney:
        sol_a, _ = reps[cid][a1]
        sol_b, _ = reps[cid][a2]
        user = ('Problem:' + chr(10) + qmap[cid] + chr(10) * 2 +
                'Solution A:' + chr(10) + sol_a[:3000] + chr(10) * 2 +
                'Solution B:' + chr(10) + sol_b[:3000] + chr(10) * 2 +
                'Which solution correctly solves the problem, A or B?')
        judge_prompts.append(tok.apply_chat_template(
            [{'role': 'system', 'content': JUDGE_SYSTEM},
             {'role': 'user', 'content': user}],
            tokenize=False, add_generation_prompt=True))

    judge_outs = llm.generate(judge_prompts, SamplingParams(
        n=args.votes, temperature=0.7, top_p=0.8, max_tokens=5, seed=123))

    winner = {}
    for (cid, a1, a2), o in zip(tourney, judge_outs):
        votes_a = votes_b = 0
        for c in o.outputs:
            t = c.text.strip().upper()
            if t.startswith('A'):
                votes_a += 1
            elif t.startswith('B'):
                votes_b += 1
        winner[cid] = a1 if votes_a >= votes_b else a2

    # 최종 답 조립: 비접전+스킵 문제는 기준 가중값, 토너먼트 문제는 승자
    final = dict(baseline)
    final.update(winner)

    gold = {r['id']: r['gold'] for r in rows}
    base_correct = sum(1 for cid in gold if baseline[cid] == gold[cid])
    final_correct = sum(1 for cid in gold if final[cid] == gold[cid])

    tourney_ids = set(cid for cid, _, _ in tourney)
    t_base_correct = sum(1 for cid in tourney_ids if baseline[cid] == gold[cid])
    t_final_correct = sum(1 for cid in tourney_ids if final[cid] == gold[cid])

    total = len(rows)
    print(chr(10) + f'[baseline weighted n={args.n}] {base_correct/total:.1%} ({base_correct}/{total})')
    print(f'[pairwise tournament] {final_correct/total:.1%} ({final_correct}/{total}), '
          f'delta={final_correct-base_correct:+d}문제 ({(final_correct-base_correct)/total*100:+.2f}%p)')
    print(f'접전 부분집합만: baseline {t_base_correct}/{len(tourney_ids)} -> '
          f'tournament {t_final_correct}/{len(tourney_ids)}')

    os.makedirs('results', exist_ok=True)
    out = {
        'n': args.n, 'margin': args.margin, 'gen_n': args.gen_n, 'votes': args.votes,
        'total': total, 'contested': len(contested), 'skipped_no_rep': skipped,
        'tourney_n': len(tourney),
        'baseline_acc': base_correct / total, 'baseline_correct': base_correct,
        'tournament_acc': final_correct / total, 'tournament_correct': final_correct,
        'delta_problems': final_correct - base_correct,
        'subset_baseline_correct': t_base_correct, 'subset_tournament_correct': t_final_correct,
    }
    json.dump(out, open('results/eval_pairwise_tournament.json', 'w'), indent=2)
    print('저장: results/eval_pairwise_tournament.json')


if __name__ == '__main__':
    main()
