"""덤프에서 제출 파일 생성 — GPU 없이 수 초. 하루 최대 건수 제출용.

사용 예:
    uv run python remote/make_submission_from_dump.py --rule weighted --n 96 --tag w96
    uv run python remote/make_submission_from_dump.py --rule majority --n 96 --tag m96
    uv run python remote/make_submission_from_dump.py --rule trim25   --n 96 --tag t96
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict

import pandas as pd


def majority(ss):
    v = [s['ans'] for s in ss if s['ans'] is not None]
    return Counter(v).most_common(1)[0][0] if v else 0


def weighted(ss, scale=2.0):
    v = [(s['ans'], s['logp']) for s in ss if s['ans'] is not None]
    if not v:
        return 0
    m = max(lp for _, lp in v)
    w = defaultdict(float)
    for a, lp in v:
        w[a] += math.exp((lp - m) * scale)
    return max(w, key=w.get)


def drop_trunc(ss):
    kept = [s for s in ss if not s['trunc']]
    return majority(kept if any(s['ans'] is not None for s in kept) else ss)


def trim25(ss):
    v = [s for s in ss if s['ans'] is not None]
    if len(v) < 4:
        return majority(ss)
    v.sort(key=lambda s: s['logp'], reverse=True)
    return majority(v[:max(2, int(len(v) * 0.75))])


def tiebreak_conf(ss):
    v = [s for s in ss if s['ans'] is not None]
    if not v:
        return 0
    cnt = Counter(s['ans'] for s in v)
    top = cnt.most_common()
    tied = [a for a, c in top if c == top[0][1]]
    if len(tied) == 1:
        return tied[0]
    best = defaultdict(lambda: -1e9)
    for s in v:
        best[s['ans']] = max(best[s['ans']], s['logp'])
    return max(tied, key=lambda a: best[a])


RULES = {'majority': majority, 'weighted': weighted, 'drop_trunc': drop_trunc,
         'trim25': trim25, 'tiebreak_conf': tiebreak_conf,
         'weighted_s4': lambda ss: weighted(ss, 4.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='results/lb_samples.jsonl')
    ap.add_argument('--rule', default='weighted', choices=sorted(RULES))
    ap.add_argument('--n', type=int, default=None, help='앞에서 n표만 사용 (없으면 전부)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--lb-csv', default='deep-learning-challenge-2026/deep_chal_math_leaderboard_filtered.csv')
    ap.add_argument('--allow-partial', action='store_true',
                    help='덤프가 전 문항을 덮지 않아도 진행 (스모크 테스트 전용)')
    args = ap.parse_args()

    rows = {json.loads(l)['id']: json.loads(l)['samples']
            for l in open(args.dump, encoding='utf-8')}
    lb = pd.read_csv(args.lb_csv)
    lb.columns = lb.columns.str.strip()

    # ⚠️ 2026-08-15 추가. 덤프에 없는 문항은 아래에서 조용히 0으로 채워진다. 그래서 덤프가
    # 중간에 죽은 상태로 이 스크립트를 돌리면 **겉보기엔 정상인 제출 파일**이 나온다 —
    # 행수도 맞고 결측·중복도 없어서 문서의 검증 절차를 그대로 통과한다.
    # (스모크 테스트에서 20문항만 덤프했더니 831행 중 811개가 0으로 채워진 채 저장됐다.)
    # 8/31에 이 파일을 그대로 제출하면 대부분이 오답이 되므로, 기본값은 '거부'로 둔다.
    missing = [pid for pid in lb['id'] if pid not in rows]
    if missing:
        msg = (f'덤프가 {len(missing)}/{len(lb)}문항을 덮지 않는다 '
               f'(예: {", ".join(map(str, missing[:3]))}). '
               f'이대로 만들면 그 문항들이 전부 0으로 채워진다.')
        if not args.allow_partial:
            raise SystemExit(
                '⛔ ' + msg + chr(10) +
                '   덤프를 끝까지 돌리거나(같은 명령 재실행 시 완료분은 건너뜀), '
                '의도한 것이면 --allow-partial 을 붙일 것.')
        print('⚠️ ' + msg + ' (--allow-partial 지정됨)')

    fn = RULES[args.rule]
    preds = []
    for pid in lb['id']:
        ss = rows.get(pid, [])
        if args.n:
            ss = ss[:args.n]
        preds.append(fn(ss) if ss else 0)

    os.makedirs('results', exist_ok=True)
    out = f'results/submission_{args.tag}.csv'
    sub = pd.DataFrame({'id': lb['id'], 'answer': preds})
    sub['answer'] = sub['answer'].astype('int64')
    sub.to_csv(out, index=False)

    # 참고: 기존 제출과 몇 문항이나 다른지 (제출 가치 판단용)
    prev = 'results/submission_n32w.csv'
    diff = ''
    if os.path.exists(prev):
        p = pd.read_csv(prev)
        merged = sub.merge(p, on='id', suffixes=('_new', '_old'))
        d = (merged['answer_new'] != merged['answer_old']).sum()
        diff = f' / 기존 제출(n32w)과 {d}문항 다름'
    print(f'저장: {out} ({len(sub)}행, rule={args.rule}, n={args.n or "all"}){diff}')


if __name__ == '__main__':
    main()
