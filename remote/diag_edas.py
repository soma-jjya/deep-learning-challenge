"""exp84 — EDAS(arXiv:2605.17333) 오프라인 진단. GPU 0.

논문 수식을 그대로 구현한다 (사전등록 experiments/prereg_exp77_decoding.md의 exp84 절):

    오답 궤적 i ∈ W:
      Δᵢ = 0                         (Nw ≤ 1)
         = −β·S                      (Nw > 1, K = 1)
         = α·S·(Iᵢ − H)/ln(Nw)       (K > 1)
      S  = (1/Nw)·Σ|Aᵢ_orig|  (오답 부분집합 평균 절대 advantage)
      Iᵢ = −ln(p_k), p_k = |C_k|/Nw ;  H = −Σ p_k ln p_k
      A_final = A_orig + sgn(Δᵢ)·min(|Δᵢ|, |A_orig|/κ)
      α=0.4, β=0.2, κ=2.0. 정답 궤적 불변.

베이스 롤아웃(483×32)을 학습 초기 GRPO 그룹의 근사로 쓴다(exp12는 베이스에서 시작).
A_orig = (r − mean)/(std+ε), r ∈ {0,1}. 민감도: 평균중심화만.

사용: PYTHONIOENCODING=utf-8 python remote/diag_edas.py
"""
import json
import math
import statistics
from collections import Counter, defaultdict

ALPHA, BETA, KAPPA = 0.4, 0.2, 2.0
EPS = 1e-8
N = 32


def grpo_adv(rs, center_only=False):
    m = sum(rs) / len(rs)
    if center_only:
        return [r - m for r in rs]
    sd = statistics.pstdev(rs)
    return [(r - m) / (sd + EPS) if sd > 0 else 0.0 for r in rs]


def edas(ans_list, gold, advs):
    """논문 수식 그대로. (A_final 리스트, 진단 dict) 반환."""
    W = [i for i, a in enumerate(ans_list) if a != gold]        # 오답 인덱스 (None 포함)
    Nw = len(W)
    out = list(advs)
    diag = {'Nw': Nw, 'K': 0, 'S': 0.0, 'delta': {}}
    if Nw == 0:
        return out, diag
    # 오답 클래스: 정규화된 최종 수치 답. 추출 실패(None)는 별도 클래스 (구현 선택, prereg 명시)
    cls = Counter('NONE' if ans_list[i] is None else ans_list[i] for i in W)
    K = len(cls)
    S = sum(abs(advs[i]) for i in W) / Nw
    H = -sum((c / Nw) * math.log(c / Nw) for c in cls.values())
    diag.update(K=K, S=S)
    for i in W:
        key = 'NONE' if ans_list[i] is None else ans_list[i]
        if Nw <= 1:
            d = 0.0
        elif K == 1:
            d = -BETA * S
        else:
            Ii = -math.log(cls[key] / Nw)
            d = ALPHA * S * (Ii - H) / math.log(Nw)
        clipped = math.copysign(min(abs(d), abs(advs[i]) / KAPPA), d) if d != 0 else 0.0
        out[i] = advs[i] + clipped
        diag['delta'][i] = (d, clipped)
    return out, diag


def main():
    rows = [json.loads(l) for l in open('results/val_samples.jsonl', encoding='utf-8')]
    print('exp84 — EDAS 오프라인 진단 · 483×%d 베이스 롤아웃 · α=%.1f β=%.1f κ=%.1f'
          % (N, ALPHA, BETA, KAPPA))
    print()

    # SC 구간 분류 (가중투표)
    def wv(ss):
        v = [(s['ans'], s['logp']) for s in ss[:N] if s.get('ans') is not None]
        if not v:
            return None
        m = max(lp for _, lp in v)
        w = defaultdict(float)
        for a, lp in v:
            w[a] += math.exp((lp - m) * 2.0)
        return max(w, key=w.get)

    typology = Counter()
    g1_active = g1_total = 0
    allwrong_K = Counter()
    mixed_stats = []          # (지배클래스 |A| 증폭률, 희소클래스 완화율)
    rec_stats = []            # 회수 구간의 지배오답 억제
    correct_changed = 0
    for r in rows:
        ss = r['samples'][:N]
        gold = r['gold']
        ans = [s.get('ans') for s in ss]
        rs = [1.0 if a == gold else 0.0 for a in ans]
        advs = grpo_adv(rs)
        fin, dg = edas(ans, gold, advs)
        nc = int(sum(rs))
        sc = wv(ss)
        has = gold in ans

        # 정답 advantage 보존 검증
        for i in range(len(ss)):
            if rs[i] == 1.0 and fin[i] != advs[i]:
                correct_changed += 1

        if nc == 0:
            typology['전원 오답 (K=%s)' % ('1' if dg['K'] == 1 else '>1')] += 1
            allwrong_K[dg['K']] += 1
            g1_total += 1
            if any(fin[i] != 0.0 for i in range(len(ss))):
                g1_active += 1
        elif nc == len(ss):
            typology['전원 정답'] += 1
        else:
            typology['혼합'] += 1
            # 지배 클래스 vs 희소 클래스의 advantage 변화
            W = [i for i, a in enumerate(ans) if a != gold]
            cls = Counter('NONE' if ans[i] is None else ans[i] for i in W)
            if len(cls) > 1:
                dom_key, dom_n = cls.most_common(1)[0]
                dom = [i for i in W if ('NONE' if ans[i] is None else ans[i]) == dom_key]
                rare = [i for i in W if ('NONE' if ans[i] is None else ans[i]) != dom_key]
                amp = (sum(abs(fin[i]) for i in dom) / len(dom)) / \
                      (sum(abs(advs[i]) for i in dom) / len(dom))
                att = (sum(abs(fin[i]) for i in rare) / len(rare)) / \
                      (sum(abs(advs[i]) for i in rare) / len(rare))
                mixed_stats.append((amp, att))
                if sc != gold and has:
                    rec_stats.append((amp, att, dom_n))

    print('그룹 유형 (483문항):')
    for k, v in typology.most_common():
        print('  %-22s %4d  (%.1f%%)' % (k, v, 100 * v / len(rows)))
    print()

    print('=' * 66)
    print('[G1] 전원-오답 그룹(exp12/24식 GRPO에서 그래디언트 0)의 회복')
    print('  전원-오답 %d그룹 중 EDAS가 0이 아닌 advantage를 만든 그룹: %d개 (%.1f%%)'
          % (g1_total, g1_active, 100 * g1_active / max(1, g1_total)))
    print('  기제: S = 오답 부분집합의 평균 |A_orig| 인데, 전원-오답이면 그룹 정규화로')
    print('        A_orig가 전부 0 → S=0 → Δ=0. 클리핑 상한 |A_orig|/κ도 0이다.')
    print('  전원-오답 그룹의 오답 클래스 수 K 분포: %s'
          % dict(sorted(allwrong_K.items())))
    print()

    print('[G2] 혼합 그룹의 재분배 (K>1인 혼합 %d그룹)' % len(mixed_stats))
    if mixed_stats:
        amp = [a for a, _ in mixed_stats]
        att = [b for _, b in mixed_stats]
        print('  지배 오답 클래스 |A| 배율: 평균 %.3f (중앙 %.3f) — 1보다 크면 벌점 증폭'
              % (sum(amp) / len(amp), sorted(amp)[len(amp) // 2]))
        print('  희소 오답 클래스 |A| 배율: 평균 %.3f (중앙 %.3f) — 1보다 작으면 벌점 완화'
              % (sum(att) / len(att), sorted(att)[len(att) // 2]))
        n_ok = sum(1 for a, b in mixed_stats if a > 1.0 and b < 1.0)
        print('  방향이 논문 의도와 일치(증폭&완화)한 그룹: %d/%d (%.1f%%)'
              % (n_ok, len(mixed_stats), 100 * n_ok / len(mixed_stats)))
    print()
    print('[④] 회수 구간(SC오답·정답존재)만 (%d그룹):' % len(rec_stats))
    if rec_stats:
        amp = [a for a, _, _ in rec_stats]
        att = [b for _, b, _ in rec_stats]
        dn = [d for _, _, d in rec_stats]
        print('  지배오답 벌점 배율 평균 %.3f · 희소 완화 배율 평균 %.3f · 지배오답 평균 %.1f표'
              % (sum(amp) / len(amp), sum(att) / len(att), sum(dn) / len(dn)))
    print('[③] 정답 궤적 advantage가 바뀐 사례: %d건 (0이어야 함)' % correct_changed)
    print()

    # 민감도: 평균중심화 advantage (std 정규화 없음)
    g1b_active = g1b_total = 0
    for r in rows:
        ss = r['samples'][:N]
        ans = [s.get('ans') for s in ss]
        rs = [1.0 if a == r['gold'] else 0.0 for a in ans]
        if sum(rs) == 0:
            g1b_total += 1
            fin, _ = edas(ans, r['gold'], grpo_adv(rs, center_only=True))
            if any(v != 0.0 for v in fin):
                g1b_active += 1
    print('민감도(평균중심화만): 전원-오답 %d그룹 중 회복 %d개 — 정규화 방식과 무관한지 확인'
          % (g1b_total, g1b_active))
    print()
    print('사전등록 게이트: G1(전원-오답 회복)과 G2(혼합 재분배 방향) **둘 다** 통과해야 GPU 파일럿.')
    print('G2만 통과면 GPU 미투입 — 이미 있던 그래디언트의 ≤%.0f%% 재분배인데(κ=%.0f 클리핑),'
          % (100 / KAPPA, KAPPA))
    print('exp12/24는 바로 그 그래디언트로 보상 곡선이 평평했다.')


if __name__ == '__main__':
    main()
