"""exp77 — First Finish Search 오프라인 진단 (GPU 0)

가설(H33, arXiv:2505.18149): n개 추론 흐름을 병렬로 띄우고 **가장 먼저 끝난 것을 그대로 채택**하면
투표보다 낫다. 짧게 끝나는 추론이 더 정확한 경향을 이용한다.

⚠️ exp06c(최단 풀이 **학습**)와 다르다. 여기는 학습이 없는 **추론 규칙**이다.
   집계 규칙 63종 중 길이를 쓴 것은 drop_trunc(잘린 표본 제외)뿐이고 최단 1개 채택은 없었다.

대리 지표: 기존 덤프의 `len`(생성 토큰 수)이 최소인 표본 = 가장 먼저 끝난 표본.
실제 wall-clock 완료 순서와 완전히 같지는 않다(배치 스케줄링·프리필 길이 영향).
**GPU를 쓸 가치가 있는지 판단하는 용도로만 쓴다.**

잘린 표본(trunc=True)은 max_tokens에 걸린 것이라 애초에 "끝나지 않은" 흐름이므로
FFS 후보에서 제외한다(어차피 길이가 최대라 최소로 뽑히지도 않는다).

사용: PYTHONIOENCODING=utf-8 python remote/diag_first_finish.py
"""
import json
import math
import random
from collections import defaultdict

SEEDS = [('42', 'results/val_samples.jsonl'),
         ('43', 'results/val_samples_s43.jsonl'),
         ('44', 'results/val_samples_s44.jsonl')]
NS = [2, 4, 8, 16, 32]
TRIALS = 200
random.seed(0)


def wvote(ss, scale=2.0):
    v = [(s['ans'], s['logp']) for s in ss if s.get('ans') is not None]
    if not v:
        return None
    m = max(lp for _, lp in v)
    w = defaultdict(float)
    for a, lp in v:
        w[a] += math.exp((lp - m) * scale)
    return max(w, key=w.get)


def ffs_pick(ss):
    """가장 먼저 끝난 표본 = 잘리지 않은 것 중 토큰 수 최소"""
    fin = [s for s in ss if not s.get('trunc')]
    if not fin:
        fin = list(ss)
    return min(fin, key=lambda s: s.get('len', 10 ** 9)).get('ans')


def main():
    print('exp77 — First Finish Search 오프라인 진단 (GPU 0)')
    print('반복 부분추출 %d회 · 시드 %d개 · 지표는 정답 문항 수(전체 483)' % (TRIALS, len(SEEDS)))
    print()

    agg = defaultdict(list)     # (n, rule, region) -> 시드별 값
    for sid, path in SEEDS:
        rows = [json.loads(l) for l in open(path, encoding='utf-8')]
        N = len(rows)
        # 구간 정의는 그 시드의 전체 n=32 가중 SC 기준
        sc32 = {r['id']: wvote(r['samples'][:32]) for r in rows}
        has = {r['id']: (r['gold'] in [s['ans'] for s in r['samples'][:32]]) for r in rows}
        reg = {}
        for r in rows:
            i = r['id']
            if sc32[i] == r['gold']:
                reg[i] = 'SC정답'
            elif has[i]:
                reg[i] = '회수가능'          # SC오답 & pass@32=1
            else:
                reg[i] = '정답없음'
        counts = defaultdict(int)
        for i in reg:
            counts[reg[i]] += 1

        print('[시드 %s] 전체 %d · SC정답 %d · 회수가능 %d · 정답없음 %d'
              % (sid, N, counts['SC정답'], counts['회수가능'], counts['정답없음']))

        for n in NS:
            hit = defaultdict(float)
            for r in rows:
                g, ss, i = r['gold'], r['samples'][:32], r['id']
                if n == 32:
                    draws = [ss]
                    reps = 1
                else:
                    draws = [random.sample(ss, n) for _ in range(TRIALS)]
                    reps = TRIALS
                a_ffs = a_rnd = a_sc = 0
                for d in draws:
                    a_ffs += (ffs_pick(d) == g)
                    a_rnd += (d[0].get('ans') == g)
                    a_sc += (wvote(d) == g)
                for rule, v in (('FFS', a_ffs), ('무작위1개', a_rnd), ('가중SC', a_sc)):
                    hit[(rule, 'ALL')] += v / reps
                    hit[(rule, reg[i])] += v / reps
            for rule in ('FFS', '무작위1개', '가중SC'):
                for rg in ('ALL', 'SC정답', '회수가능', '정답없음'):
                    agg[(n, rule, rg)].append(hit[(rule, rg)])

    print()
    print('=' * 74)
    print('%-6s%-10s%10s%10s%10s%10s' % ('n', '규칙', '전체', 'SC정답', '회수가능', '정답없음'))
    print('-' * 74)
    base32 = sum(agg[(32, '가중SC', 'ALL')]) / 3
    for n in NS:
        for rule in ('FFS', '무작위1개', '가중SC'):
            v = [sum(agg[(n, rule, rg)]) / 3 for rg in ('ALL', 'SC정답', '회수가능', '정답없음')]
            mark = ''
            if rule == 'FFS':
                d = v[0] - base32
                mark = '   %+.1f문항 vs 현행' % d
            print('%-6s%-10s%10.1f%10.1f%10.1f%10.1f%s' % (n if rule == 'FFS' else '', rule,
                                                           v[0], v[1], v[2], v[3], mark))
        print('-' * 74)

    print()
    print('시드별 FFS 전체 정답 수 (부호 일관성 확인):')
    for n in NS:
        vals = [round(x, 1) for x in agg[(n, 'FFS', 'ALL')]]
        b = [round(x, 1) for x in agg[(32, '가중SC', 'ALL')]]
        d = [round(vals[k] - b[k], 1) for k in range(3)]
        print('  n=%-3d FFS %s   vs 현행 n=32 가중SC %s   차 %s' % (n, vals, b, d))

    print()
    print('사전 등록한 판정 기준:')
    print('  · 성공: 세 시드 모두에서 현행 이상이고 평균 +1.5%p(약 +7.2문항) 이상')
    print('  · 중단: 세 시드 평균이 현행보다 낮으면 GPU로 올리지 않고 종료')


if __name__ == '__main__':
    main()
