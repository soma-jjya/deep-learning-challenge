"""계열 상관을 반영한 앙상블 — 35개 시스템을 독립 시스템으로 취급하지 않는다.

exp73에서 결합 규칙 5종이 전부 음수였는데, 그 규칙들은 시스템을 **동등한 표 하나씩**으로
취급했다. 그런데 base_s42/s43/s44는 사실상 한 가족이다. 세 표를 주면 **같은 오류를 3배 세게
세는 꼴**이다. 실제로 exp73 대조군에서 같은 계열 3개 합집합은 +4인데 이종 계열은 +14였다 —
**독립 정보의 단위가 sample이나 system이 아니라 family일 가능성**이 있다.

그래서:
  1) 먼저 계열 내부에서 시드를 합쳐 **계열당 답 하나**를 만든다
  2) 계열 간 상관행렬을 실측해 중복 정보를 확인한다
  3) 계열 신뢰도 x 다양성으로 가중해 최종 답을 정한다

⚠️ 신뢰도 가중치를 483문항 전체에서 뽑아 483문항에서 평가하면 그건 선택 후 평가다.
   **5-fold CV**로 가중치는 학습 폴드에서만 뽑고 평가는 홀드아웃 폴드에서 한다.

사용: PYTHONIOENCODING=utf-8 python remote/diag_family_ensemble.py
"""
import json
import math
import os
import sys
from collections import defaultdict

R = 'results/'
FAMILIES = {
    'base':     ['val_samples.jsonl', 'val_samples_s43.jsonl', 'val_samples_s44.jsonl'],
    'dpo':      ['val_samples_dpo.jsonl', 'val_samples_dpo_s43.jsonl', 'val_samples_dpo_s44.jsonl'],
    'teacher':  ['val_samples_teacher_full_s42.jsonl', 'val_samples_teacher_full_s43.jsonl',
                 'val_samples_teacher_full_s44.jsonl', 'val_samples_teacher_s42.jsonl',
                 'val_samples_teacher_s43.jsonl', 'val_samples_teacher_ep1_s42.jsonl'],
    'rftmask':  ['val_samples_rftmask_s42.jsonl', 'val_samples_rftmask_s43.jsonl',
                 'val_samples_rftmask_s44.jsonl'],
    'masked':   ['val_samples_masked_s42.jsonl', 'val_samples_masked_s43.jsonl',
                 'val_samples_masked_s44.jsonl'],
    'mt3072':   ['val_samples_mt3072_s42.jsonl', 'val_samples_mt3072_s43.jsonl',
                 'val_samples_mt3072_s44.jsonl'],
    'marginal': ['val_samples_marginal_s42.jsonl', 'val_samples_marginal_s43.jsonl'],
    'prompt':   ['val_samples_sp_base_s42.jsonl', 'val_samples_sp_classify_s42.jsonl',
                 'val_samples_sp_direct_s42.jsonl', 'val_samples_sp_extract_s42.jsonl',
                 'val_samples_sp_verify_s42.jsonl'],
    'temp':     ['val_samples_tp_base_t03_s42.jsonl', 'val_samples_tp_base_t07_s42.jsonl',
                 'val_samples_tp_base_t11_s42.jsonl', 'val_samples_tp_minimal_t07_s42.jsonl',
                 'val_samples_tp_nosys_t07_s42.jsonl'],
    'rank':     ['val_samples_rank_s42.jsonl'],
    'rfs3':     ['val_samples_rfs3_s42.jsonl'],
}
N = 32
SCALE = 2.0


def wvote(pairs):
    """(ans, logp) 목록 -> (답, 가중 지분)"""
    v = [(a, lp) for a, lp in pairs if a is not None]
    if not v:
        return None, 0.0
    m = max(lp for _, lp in v)
    w = defaultdict(float)
    for a, lp in v:
        w[a] += math.exp((lp - m) * SCALE)
    tot = sum(w.values())
    best = max(w, key=w.get)
    return best, w[best] / tot


def main():
    gold, fam_ans, fam_conf = {}, {}, {}
    for fam, files in FAMILIES.items():
        acc = defaultdict(list)
        got = 0
        for fn in files:
            p = R + fn
            if not os.path.exists(p):
                continue
            got += 1
            with open(p, encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    gold[r['id']] = r['gold']
                    acc[r['id']].extend((s.get('ans'), s.get('logp', 0.0))
                                        for s in r['samples'][:N])
        if not got:
            continue
        a, c = {}, {}
        for i, pairs in acc.items():
            a[i], c[i] = wvote(pairs)
        fam_ans[fam], fam_conf[fam] = a, c
        print('  %-9s 덤프 %d개 · 표본 %d개/문항' % (fam, got, got * N))

    fams = [f for f in FAMILIES if f in fam_ans]
    ids = sorted(set.intersection(*(set(fam_ans[f]) for f in fams)))
    print()
    print('계열 %d개 · 공통 문항 %d개' % (len(fams), len(ids)))
    print()

    ok = {f: {i: int(fam_ans[f][i] == gold[i]) for i in ids} for f in fams}
    print('%-9s%8s%9s' % ('계열', '정확도', '비율'))
    for f in fams:
        n = sum(ok[f].values())
        print('%-9s%8d%9.1f%%' % (f, n, 100 * n / len(ids)))
    base_n = sum(ok['base'].values())
    print()

    # ── 계열 간 상관: 오답 일치율 (중복 정보량) ──
    print('오답 일치율 — 두 계열이 동시에 틀릴 때 같은 오답을 내는 비율 (높을수록 중복):')
    print('%-9s' % '' + ''.join('%9s' % f[:8] for f in fams))
    corr = {}
    for a in fams:
        row = ''
        for b in fams:
            if a == b:
                row += '%9s' % '-'
                continue
            both = [i for i in ids if not ok[a][i] and not ok[b][i]]
            same = sum(1 for i in both if fam_ans[a][i] == fam_ans[b][i])
            v = same / len(both) if both else 0.0
            corr[(a, b)] = v
            row += '%9.2f' % v
        print('%-9s%s' % (a[:8], row))
    print()

    # ── 결합 규칙들 (5-fold CV로 가중치 학습) ──
    folds = [[ids[j] for j in range(len(ids)) if j % 5 == k] for k in range(5)]

    def evaluate(rule):
        tot = 0
        for k in range(5):
            test = folds[k]
            train = [i for i in ids if i not in set(test)]
            w = rule(train)
            for i in test:
                sc = defaultdict(float)
                for f in fams:
                    if fam_ans[f][i] is not None:
                        sc[fam_ans[f][i]] += w[f] * (fam_conf[f][i] if w.get('_useconf') else 1.0)
                if sc and max(sc, key=sc.get) == gold[i]:
                    tot += 1
        return tot

    def w_equal(train):
        return {f: 1.0 for f in fams}

    def w_rel(train):
        return {f: sum(ok[f][i] for i in train) / len(train) for f in fams}

    def w_logodds(train):
        d = {}
        for f in fams:
            p = min(max(sum(ok[f][i] for i in train) / len(train), 1e-3), 1 - 1e-3)
            d[f] = max(math.log(p / (1 - p)), 0.0)
        return d

    def w_rel_div(train):
        rel = w_rel(train)
        d = {}
        for f in fams:
            others = [corr[(f, g)] for g in fams if g != f]
            div = 1.0 - (sum(others) / len(others))
            d[f] = rel[f] * div
        return d

    def w_rel_conf(train):
        d = w_rel(train)
        d['_useconf'] = True
        return d

    rules = [
        ('계열당 1표 (동등)', w_equal),
        ('신뢰도 가중', w_rel),
        ('로그오즈 가중', w_logodds),
        ('신뢰도 x 다양성', w_rel_div),
        ('신뢰도 x 계열확신도', w_rel_conf),
    ]
    print('=' * 62)
    print('%-24s%10s%12s' % ('결합 규칙', '정답', '베이스 대비'))
    print('-' * 62)
    print('%-24s%10d%12s' % ('베이스 계열 단독 (기준)', base_n, '-'))
    for name, rule in rules:
        n = evaluate(rule)
        print('%-24s%10d%+12d' % (name, n, n - base_n))

    # 참고: 계열을 무시하고 35개 시스템을 동등하게 셀 때
    flat = 0
    for i in ids:
        sc = defaultdict(float)
        for f in fams:
            if fam_ans[f][i] is not None:
                sc[fam_ans[f][i]] += len([x for x in FAMILIES[f] if os.path.exists(R + x)])
        if sc and max(sc, key=sc.get) == gold[i]:
            flat += 1
    print('%-24s%10d%+12d' % ('(참고) 덤프수 비례 가중', flat, flat - base_n))
    print()
    print('판정: 계열 단위로 접어도 베이스를 못 넘으면, 중복 정보 제거가 문제가 아니라')
    print('      애초에 계열 간 우열을 문항별로 가릴 신호가 없다는 뜻이다.')


if __name__ == '__main__':
    main()
