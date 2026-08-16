"""두 리더보드 덤프를 섞어 제출 파일을 만든다 — GPU 불필요.

## 왜 지금 이것을 만드나
2026-08-16에 드러난 패턴: **로컬에서 진 변형이 리더보드에서 이겼다.**

| 후보 | 로컬 | 리더보드 | pass@32 |
|---|---|---|---|
| 교사증류 어댑터 | −1.79%p | **0.78820** | +1.7문항 |
| temp 1.1 | −1.04%p | **0.78700** | 유지 |
| DPO 어댑터 | −0.14%p | 0.78580 | +0.6%p |
| 기존 n32w | 기준 | 0.78459 | 기준 |
| RFT 마스킹 | −1.79%p | **0.77135** | (감소 추정) |

**pass@32를 지키거나 올린 변형은 전부 LB에서 올랐고, 아닌 하나만 떨어졌다.**
exp52b와 맞물린다 — 우리 로컬 gold는 어려운 문항에서 깨져 있어, 어려운 문제를 더 푸는
변형이 로컬에서 부당하게 벌점을 받는다.

그렇다면 우리가 가진 것 중 **pass@n이 가장 높은 후보**를 내야 한다. exp56에서 측정한
베이스+교사 합집합 pass@64가 **90.5%**로 베이스 단독 87.8%보다 훨씬 높았는데,
로컬 투표가 ±0이라 묻어뒀다. 그 로컬 판정이 바로 지금 의심받는 자다.

사용: python remote/make_mixed_submission.py --a results/lb_samples.jsonl \
        --b results/lb_samples_teacher.jsonl --ka 16 --kb 16 --tag lb_mix1616
"""
import argparse
import json
import math
import os
from collections import defaultdict

import pandas as pd


def weighted(ss, scale=2.0):
    c = defaultdict(float)
    for s in ss:
        a = s.get('ans')
        if a is not None:
            c[a] += math.exp(scale * s.get('logp', 0.0))
    return max(c, key=c.get) if c else None


def load(path):
    return {json.loads(l)['id']: json.loads(l)['samples'] for l in open(path, encoding='utf-8')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='정책 A 덤프 (보통 베이스)')
    ap.add_argument('--b', required=True, help='정책 B 덤프 (보통 어댑터)')
    ap.add_argument('--ka', type=int, default=16)
    ap.add_argument('--kb', type=int, default=16)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--lb-csv',
                    default='deep-learning-challenge-2026/deep_chal_math_leaderboard_filtered.csv')
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    lb = pd.read_csv(args.lb_csv)
    lb.columns = lb.columns.str.strip()

    missing = [p for p in lb['id'] if p not in A or p not in B]
    if missing:
        raise SystemExit(f'⛔ 두 덤프가 {len(missing)}문항을 덮지 않는다 '
                         f'(예: {", ".join(map(str, missing[:3]))}). '
                         '누락분이 0으로 채워지면 겉보기엔 정상인 오답 파일이 된다.')

    preds = []
    for pid in lb['id']:
        preds.append(weighted(A[pid][:args.ka] + B[pid][:args.kb]) or 0)

    os.makedirs('results', exist_ok=True)
    out = f'results/submission_{args.tag}.csv'
    sub = pd.DataFrame({'id': lb['id'], 'answer': preds})
    sub['answer'] = sub['answer'].astype('int64')
    sub.to_csv(out, index=False)

    prev = 'results/submission_n32w.csv'
    diff = ''
    if os.path.exists(prev):
        p = pd.read_csv(prev)
        m = sub.merge(p, on='id', suffixes=('_new', '_old'))
        diff = f' / 기존 제출(n32w)과 {(m["answer_new"] != m["answer_old"]).sum()}문항 다름'
    zeros = int((sub['answer'] == 0).sum())
    print(f'저장: {out} ({len(sub)}행, A{args.ka}+B{args.kb}){diff}')
    print(f'  0 예측 {zeros}문항 (기준 제출은 20개 — 수십 개면 덤프 누락 의심)')


if __name__ == '__main__':
    main()
