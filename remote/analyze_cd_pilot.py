"""exp80a 분석 — anti-expert CD 파일럿의 사전등록 GO 조건 판정 (GPU 0).

사전등록(prereg_exp77_decoding.md)에 못 박은 조건:
  · 회수 구간에서 **gold 득표 지분 ↑ 그리고 지배오답 지분 ↓**
  · 전체 순정확도 **≥ +2문항 / 100**
  · base정답 구간 **harm ≤ 2**
셋을 모두 만족해야 483문항으로 확대한다.

λ=0은 **같은 코드 경로로 돌린 내장 대조군**이다 (base 스택으로 정확히 환원된다).
따라서 비교는 vLLM 덤프가 아니라 **λ=0 대조군**과 한다 — 디코딩 구현·토큰 한도 차이를
교란 변수로 들이지 않기 위해서다(exp58 이후 강제한 설계).

사용: PYTHONIOENCODING=utf-8 python remote/analyze_cd_pilot.py [--dump results/cd_pilot.jsonl]
"""
import argparse
import json
import math
from collections import Counter, defaultdict


def wvote(ss, scale=2.0):
    v = [(s['ans'], s['logp']) for s in ss if s.get('ans') is not None]
    if not v:
        return None, 0.0, 0.0, 0
    m = max(lp for _, lp in v)
    w = defaultdict(float)
    for a, lp in v:
        w[a] += math.exp((lp - m) * scale)
    tot = sum(w.values())
    order = sorted(w, key=w.get, reverse=True)
    return order[0], w, tot, len(w)


def shares(ss, gold):
    """(gold 지분, 지배오답 지분, 고유답 수)"""
    _, w, tot, nuniq = wvote(ss)
    if not tot:
        return 0.0, 0.0, 0
    g = w.get(gold, 0.0) / tot
    wrong = [v for k, v in w.items() if k != gold]
    d = (max(wrong) / tot) if wrong else 0.0
    return g, d, nuniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='results/cd_pilot.jsonl')
    ap.add_argument('--strata', default='experiments/exp80a_strata60.json')
    ap.add_argument('--val', default='results/val_samples.jsonl')
    ap.add_argument('--lams', type=float, nargs='+', default=None,
                    help='이 λ들만 비교 (부분 실행 중 완료된 것만 보고 싶을 때)')
    args = ap.parse_args()

    gold = {}
    for line in open(args.val, encoding='utf-8'):
        d = json.loads(line)
        gold[d['id']] = d['gold']
    strata = json.load(open(args.strata, encoding='utf-8'))
    where = {}
    for k, v in strata.items():
        for i in v:
            where[i] = k

    data = defaultdict(dict)   # lam -> id -> samples
    partial = 0
    for line in open(args.dump, encoding='utf-8'):
        d = json.loads(line)
        if d.get('partial'):
            partial += 1
            continue
        data[d['lam']][d['id']] = d['samples']

    if args.lams:
        data = {k: v for k, v in data.items() if k in set(args.lams)}
    lams = sorted(data)
    if not lams:
        print('결과 없음')
        return
    # 모든 λ가 공통으로 가진 문항에서만 비교한다 (부분 실행 대비)
    common = set.intersection(*(set(data[l]) for l in lams))
    common = sorted(common)
    print('λ = %s · 공통 문항 %d개%s' % (lams, len(common),
                                    (' (미완성 %d건 제외)' % partial) if partial else ''))
    for l in lams:
        print('   λ=%.1f 완료 문항 %d개' % (l, len(data[l])))
    print()
    cnt = Counter(where.get(i, '?') for i in common)
    print('구간 구성: ' + ' · '.join('%s %d' % (k, v) for k, v in cnt.items()))
    print()

    NAMES = {'recoverable': '회수가능(base오답·정답존재)',
             'low_margin': 'base정답·저마진', 'high_margin': 'base정답·고마진'}

    base_l = lams[0]
    if base_l != 0.0:
        print('⚠️ λ=0 대조군이 없다 — 아래 비교는 λ=%.1f 기준이다' % base_l)

    print('%-26s%8s' % ('구간', '문항') + ''.join('%12s' % ('λ=%.1f' % l) for l in lams))
    print('-' * (34 + 12 * len(lams)))
    acc = {l: {} for l in lams}
    for key in ['recoverable', 'low_margin', 'high_margin']:
        sub = [i for i in common if where.get(i) == key]
        if not sub:
            continue
        row = []
        for l in lams:
            k = sum(1 for i in sub if wvote(data[l][i])[0] == gold[i])
            acc[l][key] = k
            row.append(k)
        print('%-26s%8d' % (NAMES[key], len(sub)) + ''.join('%12d' % v for v in row))
    tot = []
    for l in lams:
        k = sum(1 for i in common if wvote(data[l][i])[0] == gold[i])
        acc[l]['ALL'] = k
        tot.append(k)
    print('%-26s%8d' % ('전체', len(common)) + ''.join('%12d' % v for v in tot))
    print('%-26s%8s' % ('  └ 대조군 대비', '') +
          ''.join('%12s' % ('%+d' % (acc[l]['ALL'] - acc[base_l]['ALL'])) for l in lams))
    print()

    # ── 득표 지분 (정확도가 안 변해도 분포가 움직였는지 본다 — exp69의 교훈) ──
    print('회수 가능 구간의 득표 지분 (정확도보다 민감한 지표)')
    print('%-26s' % '' + ''.join('%12s' % ('λ=%.1f' % l) for l in lams))
    sub = [i for i in common if where.get(i) == 'recoverable']
    if sub:
        for label, idx in (('gold 득표 지분', 0), ('지배오답 지분', 1), ('고유 답 개수', 2)):
            vals = []
            for l in lams:
                v = [shares(data[l][i], gold[i])[idx] for i in sub]
                vals.append(sum(v) / len(v))
            fmt = '%12.4f' if idx < 2 else '%12.2f'
            print('%-26s' % label + ''.join(fmt % v for v in vals))
    print()

    # ── 사전등록 GO 판정 ──
    print('=' * 62)
    print('사전등록 GO 조건 판정')
    for l in lams:
        if l == base_l:
            continue
        g0 = sum(shares(data[base_l][i], gold[i])[0] for i in sub) / max(1, len(sub))
        g1 = sum(shares(data[l][i], gold[i])[0] for i in sub) / max(1, len(sub))
        d0 = sum(shares(data[base_l][i], gold[i])[1] for i in sub) / max(1, len(sub))
        d1 = sum(shares(data[l][i], gold[i])[1] for i in sub) / max(1, len(sub))
        net = acc[l]['ALL'] - acc[base_l]['ALL']
        bc = [i for i in common if where.get(i) in ('low_margin', 'high_margin')]
        harm = sum(1 for i in bc
                   if wvote(data[base_l][i])[0] == gold[i] and wvote(data[l][i])[0] != gold[i])
        c1 = g1 > g0 and d1 < d0
        c2 = net >= 2
        c3 = harm <= 2
        print('  λ=%.1f : gold지분 %.4f→%.4f %s · 지배오답 %.4f→%.4f %s · 순 %+d %s · harm %d %s'
              % (l, g0, g1, '↑' if g1 > g0 else '↓', d0, d1, '↓' if d1 < d0 else '↑',
                 net, 'OK' if c2 else 'X', harm, 'OK' if c3 else 'X'))
        print('        → %s' % ('**GO** (483문항으로 확대)' if (c1 and c2 and c3)
                                else 'NO-GO (조건 %s 미달)'
                                % ', '.join(n for n, c in
                                            (('지분', c1), ('순이득', c2), ('harm', c3)) if not c)))
    print()
    print('※ 100문항 파일럿이라 순이득 ±2문항은 표본 노이즈와 구분이 어렵다.')
    print('   지분 지표를 함께 보는 이유가 그것이다 — 정확도가 안 변해도 분포는 움직일 수 있다.')


if __name__ == '__main__':
    main()
