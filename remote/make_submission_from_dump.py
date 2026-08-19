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

# ⚠️ 2026-08-19 추가 (8/31 신뢰성 감사에서 발견). 아래 astype('int64')는 답이 int64 범위를
# 벗어나면 터지거나 값을 손상시킨다. 그런데 **실제 덤프에 그런 답이 있고, 투표에서 이기기도 한다.**
#   · 전체 표본 795,682개 중 int64 초과 582개(0.073%), 최대 2,937자리
#   · n=32에서 규칙이 실제로 그 값을 채택한 사례 6건, n=8에서는 66건
# 원인은 잘린 풀이(1.7~1.9%)에서 추출기가 마지막 숫자를 주워 오는 경로다.
# 2,000자리 정수는 어떤 경우에도 정답이 아니므로, 범위를 벗어나면 **범위 안 후보로 다시 집계**한다.
# (문항당 정상 표본이 하나도 없을 때만 0으로 간다 — 그 경우도 로그에 남긴다.)
INT64_MAX = 2 ** 63 - 1


def apply_rule_safe(fn, ss):
    """규칙을 적용하되 int64를 벗어나는 답은 채택하지 않는다. (선택된 답, 사유) 반환"""
    a = fn(ss)
    if a is None:
        return 0, 'none'
    if abs(int(a)) <= INT64_MAX:
        return int(a), None
    sane = [s for s in ss if s.get('ans') is not None and abs(int(s['ans'])) <= INT64_MAX]
    if not sane:
        return 0, 'all_oversized'
    b = fn(sane)
    if b is None or abs(int(b)) > INT64_MAX:
        return 0, 'fallback_failed'
    return int(b), 'oversized_%dd' % len(str(abs(int(a))))


def main():
    import pandas as pd      # 규칙·가드는 pandas 없이도 임포트 가능해야 한다
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
    preds, notes = [], []
    short = []          # 요청한 n보다 표본이 적은 문항
    empty = []          # 표본이 아예 없는 문항
    nsamp_min = 10 ** 9
    for pid in lb['id']:
        ss = rows.get(pid, [])
        if not ss:
            empty.append(pid)
            preds.append(0)
            continue
        if args.n:
            if len(ss) < args.n:
                short.append((pid, len(ss)))
            ss = ss[:args.n]
        nsamp_min = min(nsamp_min, len(ss))
        a, why = apply_rule_safe(fn, ss)
        preds.append(a)
        if why:
            notes.append((pid, why))

    # ── 제출 전 자동 점검 (문서 체크리스트를 코드로 옮긴 것) ──
    if empty:
        print('⚠️ 표본이 전혀 없는 문항 %d개 → 0으로 채움: %s'
              % (len(empty), ', '.join(map(str, empty[:5]))))
    if short:
        print('⚠️ 요청 n=%d보다 표본이 적은 문항 %d개 (최소 %d개) → 있는 만큼만 집계: %s'
              % (args.n, len(short), min(k for _, k in short),
                 ', '.join('%s(%d)' % t for t in short[:5])))
    over = [t for t in notes if t[1].startswith('oversized')]
    if over:
        print('⚠️ int64를 벗어나는 답이 채택될 뻔해 범위 안 후보로 재집계한 문항 %d개: %s'
              % (len(over), ', '.join('%s[%s]' % t for t in over[:5])))
    bad = [t for t in notes if t[1] in ('all_oversized', 'fallback_failed', 'none')]
    if bad:
        print('⚠️ 유효한 답을 못 만들어 0으로 둔 문항 %d개: %s'
              % (len(bad), ', '.join('%s[%s]' % t for t in bad[:5])))
    nzero = sum(1 for v in preds if v == 0)
    print('점검: 행 %d · 0인 답 %d개(%.2f%%) · 문항당 최소 표본 %d'
          % (len(preds), nzero, 100.0 * nzero / max(1, len(preds)),
             0 if nsamp_min == 10 ** 9 else nsamp_min))
    if nzero > max(5, 0.02 * len(preds)):
        print('⛔ 0인 답이 비정상적으로 많다 — 덤프 누락을 의심할 것 (제출 전 반드시 확인)')

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
        if len(merged):      # 최종 test처럼 id 교집합이 없으면 비교 자체가 무의미
            d = (merged['answer_new'] != merged['answer_old']).sum()
            diff = f' / 기존 제출(n32w)과 {d}문항 다름 (공통 {len(merged)}문항)'
    print(f'저장: {out} ({len(sub)}행, rule={args.rule}, n={args.n or "all"}){diff}')


if __name__ == '__main__':
    main()
