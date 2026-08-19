"""exp79 — Prefix-Confidence test-time scaling 오프라인 검증 (GPU 0)

가설(H34, arXiv:2507.18122): 완성된 풀이 여러 개를 확신도로 고르는 대신,
**짧은 접두(32토큰)만 k개 생성해 확신도가 가장 좋은 하나만 끝까지 이어 쓴다.**

⚠️ exp62와 다른 점: exp62는 확신도에게 "A가 맞나 B가 맞나"를 물었다(판별력 AUC 0.610).
   여기서는 "어느 시작점에 예산을 더 쓸까"만 묻는다 — 요구 판별 난도가 더 낮다.

이 진단이 **충실한 시뮬레이션인 이유**: 우리 덤프의 32개 표본은 같은 온도에서 독립으로 뽑은
완전한 흐름이다. 각 표본의 앞 P토큰을 "접두"로 보고, k개 중 접두 확신도가 최고인 것을 골라
**그 표본의 최종 답을 채택**하면, 실제 방법이 하는 일(이긴 접두만 이어 씀)과 같다.
따라서 GPU 없이도 이 방법의 성능을 거의 그대로 잴 수 있다.

측정:
  1) 접두 확신도 ↔ 최종 정오의 문항내 판별력(AUC) — 접두 길이 8/16/32/64/128
  2) k ∈ {2,4,8,16,32}에서 접두 최고를 골랐을 때 정확도 vs 무작위 1개 vs 가중 SC
  3) 연산량 대비: 접두 k개 + 완전 생성 1개 는 SC n=32보다 훨씬 싸다

사용: PYTHONIOENCODING=utf-8 python remote/diag_prefix_confidence.py
"""
import json
import math
import random
from collections import defaultdict

PATH = 'results/val_tok_s42.jsonl'
PREFIX_LENS = [8, 16, 32, 64, 128]
KS = [2, 4, 8, 16, 32]
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


def main():
    rows = [json.loads(l) for l in open(PATH, encoding='utf-8')]
    N = len(rows)
    print('exp79 — Prefix-Confidence 오프라인 검증 (GPU 0)')
    print('문항 %d · 표본 32 · 시드 42 (토큰 단위 logprob 덤프)' % N)
    print()

    # 접두 확신도 미리 계산
    for r in rows:
        for s in r['samples']:
            t = s['tlp']
            for P in PREFIX_LENS:
                s['p%d' % P] = sum(t[:P]) / max(1, len(t[:P]))

    # ── 1) 문항내 판별력 (AUC) ──
    print('접두 확신도 ↔ 최종 정오, 문항내 쌍 비교 AUC (0.5 = 무신호):')
    print('  %-14s%10s%12s' % ('신호', 'AUC', '비교 쌍 수'))
    sigs = [('접두 %d토큰' % P, 'p%d' % P) for P in PREFIX_LENS]
    sigs.append(('전체 평균(현행)', 'logp'))
    for name, key in sigs:
        win = tie = tot = 0
        for r in rows:
            g = r['gold']
            c = [s for s in r['samples'] if s.get('ans') == g]
            w = [s for s in r['samples'] if s.get('ans') is not None and s['ans'] != g]
            for a in c:
                for b in w:
                    tot += 1
                    if a[key] > b[key]:
                        win += 1
                    elif a[key] == b[key]:
                        tie += 1
        auc = (win + 0.5 * tie) / tot if tot else 0.5
        print('  %-14s%10.3f%12d' % (name, auc, tot))
    print()

    # ── 2) k개 접두 중 최고를 골라 그 흐름의 최종 답 채택 ──
    print('=' * 70)
    print('k개 접두 중 최고 하나만 끝까지 씀 (정답 문항 수 / %d)' % N)
    print('  %-4s%12s%12s%12s%12s' % ('k', '접두32', '접두128', '무작위1개', '가중SC'))
    base = sum(1 for r in rows if wvote(r['samples']) == r['gold'])
    for k in KS:
        acc = defaultdict(float)
        for r in rows:
            g = r['gold']
            ss = r['samples']
            draws = [ss] if k == 32 else [random.sample(ss, k) for _ in range(TRIALS)]
            reps = len(draws)
            a32 = a128 = arnd = asc = 0
            for d in draws:
                a32 += (max(d, key=lambda s: s['p32'])['ans'] == g)
                a128 += (max(d, key=lambda s: s['p128'])['ans'] == g)
                arnd += (d[0]['ans'] == g)
                asc += (wvote(d) == g)
            acc['p32'] += a32 / reps
            acc['p128'] += a128 / reps
            acc['rnd'] += arnd / reps
            acc['sc'] += asc / reps
        print('  %-4d%12.1f%12.1f%12.1f%12.1f' % (k, acc['p32'], acc['p128'], acc['rnd'], acc['sc']))
    print()
    print('현행 스택 (가중 SC n=32) = %d문항' % base)
    print()
    print('연산량 참고: 접두 k=8이면 8x32토큰 + 완전생성 1개 ≈ 완전생성 1.7개분.')
    print('             현행 SC n=32는 완전생성 32개분이다 (약 19배).')
    print()
    print('사전 등록한 판정 기준:')
    print('  · 접두 AUC가 0.55 미만이면 신호 없음으로 보고 GPU 미투입')
    print('  · 정확도가 현행을 넘지 못하면, 같은 연산량 구간에서 이득이 있는지만 별도로 본다')


if __name__ == '__main__':
    main()
