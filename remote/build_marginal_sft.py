"""H31 — 경계 난이도 구간만 겨냥한 SFT 데이터. GPU 불필요.

## 왜 이것이 학습 7전과 다른가
우리 SFT 데이터는 "베이스가 푼 문제의 정답 풀이 전부"였다. 그런데 실측 분포를 보면
4표본 중 **4개 다 맞히는 문제가 54.7%**다. 즉 학습 데이터의 절반 이상이 **모델이 이미
완벽히 아는 문제**였고, 그걸 다시 먹이는 것은 신호가 없을 뿐 아니라 모델을 **기존 행동
쪽으로 더 날카롭게** 만든다 — 어려운 문제에서는 기존 *오답* 쪽으로 날카로워진다는 뜻이다.
SFT가 4번 연속 해로웠던 기제로 이것이 설명된다.

반대편도 마찬가지다. 교사증류(exp53c)는 **0/32(전혀 못 푸는)** 문제만 겨냥했다.
그건 능력 부재 구간이라 외부 지식을 주입해야 하고, 실제로 pass@32는 올랐지만 투표는 졌다.

**한 번도 안 겨냥한 것이 가운데다** — 1~3/4로 *가끔* 맞히는 1,464문항(24.4%).
이 구간이 우리 병목의 정확한 좌표다: 정답이 이미 표본에 나타나지만 표결에서 밀린다.
확률질량을 조금만 밀어주면 표차가 뒤집힌다.

## 정직한 유보
이 **문항 집합 자체**는 exp47 DPO가 이미 썼다(선호쌍 1,464개 = 같은 숫자).
다만 DPO는 chosen을 올리고 rejected를 내리는 **상대 순위** 학습이고, 여기서는 정답 경로를
직접 모방시키는 **SFT**다. 같은 데이터에 다른 개입이다. DPO는 rewards/accuracies 0.60으로
신호가 약했고 lr 5e-6·beta 0.3의 매우 보수적인 설정이었다.

사용: python remote/build_marginal_sft.py --lo 1 --hi 3
"""
import argparse
import json
import os
from collections import defaultdict

BS = chr(92)
SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/verifier.jsonl')
    ap.add_argument('--lo', type=int, default=1, help='정답 표본 수 하한(포함)')
    ap.add_argument('--hi', type=int, default=3, help='정답 표본 수 상한(포함)')
    ap.add_argument('--max-per-problem', type=int, default=2,
                    help='문항당 최대 풀이 수 — 쉬운 문항이 표본 수로 다시 지배하지 않게')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    by_id = defaultdict(list)
    for line in open(args.src, encoding='utf-8'):
        d = json.loads(line)
        by_id[d['id']].append(d)

    hist = defaultdict(int)
    rows = []
    n_prob = 0
    for pid, items in by_id.items():
        pos = [d for d in items if d.get('label') in (1, True, 'correct', 'yes', 'Yes')]
        hist[len(pos)] += 1
        if not (args.lo <= len(pos) <= args.hi):
            continue
        n_prob += 1
        # 짧은 풀이 우선 — exp06c의 관찰(긴 풀이는 3B에 역효과)을 따르되,
        # 여기서 짧음은 '헤매지 않은 경로'의 대리 지표다
        pos.sort(key=lambda d: len(d['solution']))
        for d in pos[:args.max_per_problem]:
            rows.append({'id': pid, 'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': d['question']},
                {'role': 'assistant', 'content': d['solution']},
            ]})

    print(f'{args.src}: 문항 {len(by_id)}개')
    print('정답 표본 수 분포:')
    tot = len(by_id)
    for k in sorted(hist):
        mark = ' ←경계' if args.lo <= k <= args.hi else ''
        print(f'  {k}개 정답: {hist[k]:5d}문항 ({hist[k]/tot:5.1%}){mark}')
    print()
    print(f'선택 구간 [{args.lo},{args.hi}]: {n_prob}문항 → 풀이 {len(rows)}개'
          f' (문항당 최대 {args.max_per_problem})')

    out = args.out or f'data/sft_marginal_{args.lo}{args.hi}.jsonl'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    print(f'저장: {out}')

    # 검증셋 오염 확인 — verifier.jsonl은 val을 제외하고 만들어졌어야 한다
    import pandas as pd
    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val = df.sample(500, random_state=123)
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    if os.path.exists(bad):
        val = val[~val['id'].isin(set(pd.read_csv(bad)['id']))]
    leak = set(r['id'] for r in rows) & set(val['id'])
    print(f'검증셋 483문항과의 겹침: {len(leak)}개'
          + ('  ⚠️ 오염! 평가가 무효가 된다' if leak else '  ✅ 없음'))

    print()
    print('다음: train_qlora.py --data-path ' + out + ' --epochs 2 --output-dir outputs/qlora_marginal')


if __name__ == '__main__':
    main()
