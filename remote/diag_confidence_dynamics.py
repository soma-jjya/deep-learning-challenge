"""exp85 — CDG(arXiv:2605.25244) + Consilience(arXiv:2608.09898) 오프라인 진단. GPU 0.

신호 = 확신도의 **변화량**. exp62(절대값 가중)·exp79(head/tail 각각의 판별력)와 다르며,
사전등록(prereg exp85 절)에 경계와 게이트를 고정했다.

    C_head = exp(앞 P% 토큰 평균 logprob),  C_tail = exp(뒤 P% 평균)
    ΔC = C_tail − C_head
    CDG:        s = C̄ + β·ΔC,  R(a) = |T_a|^α · mean_{i∈T_a}(s_i)     (논문 기본 P=10%, α=0.5)
    Consilience: S = C_final − α_c·C_initial   (앞 5% 건너뛴 [5%,15%)가 initial)

⚠️ 토큰 덤프는 seed 42 하나뿐 — 양성이어도 단일 시드 값이다(exp48 원칙).

사용: PYTHONIOENCODING=utf-8 python remote/diag_confidence_dynamics.py
"""
import json
import math
import random
from collections import defaultdict

random.seed(0)
PATH = 'results/val_tok_s42.jsonl'
PS = [0.10, 0.20, 0.25]


def winconf(t, a, b):
    """토큰 logprob 리스트 t의 [a,b) 구간(비율) 확신도 = exp(평균 logprob)"""
    n = len(t)
    i, j = int(a * n), max(int(a * n) + 1, int(b * n))
    seg = t[i:j]
    return math.exp(sum(seg) / len(seg))


def paired_auc(rows, fn):
    win = tie = tot = 0
    for r in rows:
        g = r['gold']
        c = [s for s in r['samples'] if s.get('ans') == g]
        w = [s for s in r['samples'] if s.get('ans') is not None and s['ans'] != g]
        for a in c:
            va = fn(a)
            for b in w:
                vb = fn(b)
                tot += 1
                if va > vb:
                    win += 1
                elif va == vb:
                    tie += 1
    return (win + 0.5 * tie) / tot if tot else 0.5


def main():
    rows = [json.loads(l) for l in open(PATH, encoding='utf-8')]
    N = len(rows)
    print('exp85 — 확신도 변화량 진단 · %d문항 × 32 · seed 42 단일 (한계 명시)' % N)
    print()

    # 미리 계산
    for r in rows:
        for s in r['samples']:
            t = s['tlp']
            s['cbar'] = math.exp(s['logp'])
            for P in PS:
                s['h%d' % int(P * 100)] = winconf(t, 0.0, P)
                s['t%d' % int(P * 100)] = winconf(t, 1.0 - P, 1.0)
                s['d%d' % int(P * 100)] = s['t%d' % int(P * 100)] - s['h%d' % int(P * 100)]
            s['cons'] = s['t10'] - winconf(t, 0.05, 0.15)     # Consilience (α_c=1)

    # ── 1단계: 판별력 ──
    print('[1] 문항내 대응 AUC (0.5 = 무신호):')
    print('  %-26s%8s' % ('신호', 'AUC'))
    sigs = []
    for P in PS:
        k = int(P * 100)
        sigs += [('head %d%%' % k, 'h%d' % k), ('tail %d%%' % k, 't%d' % k),
                 ('ΔC = tail−head %d%%' % k, 'd%d' % k)]
    sigs += [('Consilience S (α=1)', 'cons'), ('전체 평균 (현행)', 'cbar')]
    aucs = {}
    for name, key in sigs:
        a = paired_auc(rows, lambda s, key=key: s[key])
        aucs[key] = a
        mark = ' ◀' if key.startswith('d') or key == 'cons' else ''
        print('  %-26s%8.3f%s' % (name, a, mark))
    best_tail = max(aucs['t%d' % int(P * 100)] for P in PS)
    best_delta = max([aucs['d%d' % int(P * 100)] for P in PS] + [aucs['cons']])
    print()
    print('  G1 판정: max AUC(ΔC계열) = %.3f vs max AUC(tail 단독) = %.3f  (기준: +0.02 이상)'
          % (best_delta, best_tail))
    g1 = best_delta > best_tail + 0.02
    print('  → G1 %s' % ('통과' if g1 else '실패 — 변화량은 tail 절대값의 재포장'))
    print()

    # 정오답 ΔC 평균 + 문항 부트스트랩 CI (P=10%)
    per_prob = []
    for r in rows:
        g = r['gold']
        c = [s['d10'] for s in r['samples'] if s.get('ans') == g]
        w = [s['d10'] for s in r['samples'] if s.get('ans') is not None and s['ans'] != g]
        if c and w:
            per_prob.append((sum(c) / len(c), sum(w) / len(w)))
    diffs = []
    for _ in range(2000):
        bs = [per_prob[random.randrange(len(per_prob))] for _ in range(len(per_prob))]
        diffs.append(sum(a - b for a, b in bs) / len(bs))
    diffs.sort()
    print('[2] ΔC(10%%) 평균: 정답 %.4f vs 오답 %.4f · 차이 부트스트랩 95%% CI [%.4f, %.4f] (%d문항)'
          % (sum(a for a, _ in per_prob) / len(per_prob),
             sum(b for _, b in per_prob) / len(per_prob),
             diffs[50], diffs[1949], len(per_prob)))
    print()

    # ── 2단계: 집계 (1차 = 논문 기본값 고정) ──
    def agg(r, rule, beta=1.0, alpha=0.5, ac=1.0):
        pool = defaultdict(list)
        for s in r['samples']:
            if s.get('ans') is None:
                continue
            pool[s['ans']].append(s)
        if not pool:
            return None
        sc = {}
        for a, ts in pool.items():
            n = len(ts)
            if rule == 'majority':
                sc[a] = n
            elif rule == 'weighted':                     # 현행 스택
                m = max(x['logp'] for t2 in pool.values() for x in t2)
                sc[a] = sum(math.exp((t['logp'] - m) * 2.0) for t in ts)
            elif rule == 'sqrt_conf':                    # √count 효과 분리 대조군
                sc[a] = (n ** alpha) * (sum(t['cbar'] for t in ts) / n)
            elif rule == 'cdg':
                sc[a] = (n ** alpha) * (sum(t['cbar'] + beta * t['d10'] for t in ts) / n)
            elif rule == 'consilience':
                sc[a] = sum(t['t10'] - ac * winconf(t['tlp'], 0.05, 0.15) for t in ts) / n
            elif rule == 'consilience_sqrt':
                sc[a] = (n ** alpha) * (sum(t['cons'] for t in ts) / n)
        return max(sc, key=sc.get)

    print('[3] 집계 정확도 (전체 483, 논문 기본값 — 사후 최적화 없음):')
    base_w = sum(1 for r in rows if agg(r, 'weighted') == r['gold'])
    for rule, label in [('majority', 'majority'), ('weighted', '현행 가중 (기준)'),
                        ('sqrt_conf', 'count^0.5 · C̄  (대조)'),
                        ('cdg', 'CDG (P=10, α=0.5, β=1)'),
                        ('consilience', 'Consilience (α_c=1)'),
                        ('consilience_sqrt', 'Consilience × count^0.5')]:
        k = sum(1 for r in rows if agg(r, rule) == r['gold'])
        print('  %-28s %d  (%+d)' % (label, k, k - base_w))
    print()

    # nested CV: 안쪽 폴드에서 (rule, β/α_c) 선택 → 바깥 폴드 평가
    print('[4] nested CV 배포 순이득 (5-fold, 안쪽에서만 파라미터 선택):')
    grid = [('cdg', b, 1.0) for b in (0.5, 1.0, 2.0)] + \
           [('consilience', 1.0, a) for a in (0.5, 1.0, 2.0)] + \
           [('consilience_sqrt', 1.0, 1.0), ('sqrt_conf', 1.0, 1.0)]
    idx = list(range(N))
    folds = [idx[k::5] for k in range(5)]
    net = 0
    for k in range(5):
        outer = set(folds[k])
        inner = [i for i in idx if i not in outer]
        best, bacc = None, -1
        for rule, b, a in grid:
            acc = sum(1 for i in inner if agg(rows[i], rule, beta=b, ac=a) == rows[i]['gold'])
            if acc > bacc:
                bacc, best = acc, (rule, b, a)
        rule, b, a = best
        got = sum(1 for i in outer if agg(rows[i], rule, beta=b, ac=a) == rows[i]['gold'])
        ref = sum(1 for i in outer if agg(rows[i], 'weighted') == rows[i]['gold'])
        net += got - ref
        print('  fold %d: 선택 %-18s → 바깥 %+d (선택 %d vs 현행 %d)'
              % (k, str(best), got - ref, got, ref))
    print('  → G2: nested CV 순이득 합계 %+d문항 (기준: > 0)' % net)
    print()
    print('사전등록 게이트: G1 %s · G2 %s' % ('통과' if g1 else '실패', '통과' if net > 0 else '실패'))
    print('⚠️ seed 42 단일 — 어떤 결론도 s43/44 토큰 덤프 없이 채택하지 않는다.')


if __name__ == '__main__':
    main()
