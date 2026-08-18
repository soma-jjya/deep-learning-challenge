"""문제 유형별 전문화가 실재하는가 — 시스템 상보성(+14)이 "전문화"인지 "무작위"인지 판별.

배경(exp73): 35개 시스템의 합집합 오라클이 400/483(82.8%)로 최고 단일 시스템(370)을 훨씬
넘었고, 같은 계열 시드끼리 합치면 +4인데 이종 계열끼리는 +14였다. 즉 **이종 계열 간
상보성은 시드 노이즈 바닥의 3.5배**로 실재한다. 그런데 exp73에서 시도한 결합 규칙 5종
(시스템 다수결/최고 득표율/가중 득표율/2대1 역전/베이스 저확신 게이트)은 전부 -2~-5였다.

그 규칙들의 공통점: **전부 생성 이후 신호(득표·logprob·합의)를 봤다.** 그리고 그 신호들이
왜 안 되는지는 이미 안다 — 틀렸는데도 확신하는 경우가 많아 판별력이 0.6 벽에 걸린다.

그래서 여기서는 모델의 확신을 아예 보지 않고 **문제 텍스트만으로** 묻는다:
   "이 유형의 문제에서는 특정 계열이 베이스보다 체계적으로 낫거나 못한가?"

  A. 그냥 무작위로 다른 문제를 맞힌다      -> 라우팅 불가능, 여기서 닫는다
  B. 특정 유형에서 실제로 더 강하다        -> problem-conditioned router로 진행

주의: 이 진단의 핵심은 **시드 대응**이다(exp48 원칙). 시드 42에서 "정수론에서 교사가 +6"이
나오는 것은 아무 의미가 없다 — 자유도가 유형 수 x 계열 수만큼 있으므로 우연히 좋아 보이는
칸이 반드시 나온다. 그래서 계열마다 시드 3개를 **베이스의 같은 시드와 짝지어** 효용을 재고,
**세 시드에서 부호가 일치하는 칸만** 신호로 인정한다.
그리고 최대 통계량에 대한 순열검정으로 "가장 좋아 보이는 칸"이 우연으로 나올 확률을 잰다.

사용: PYTHONIOENCODING=utf-8 python remote/diag_specialist_localization.py
"""
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

csv.field_size_limit(10 ** 7)
random.seed(0)

TRAIN = 'deep-learning-challenge-2026/deep_chal_math_train.csv'
N = 32

# 계열: 이름 -> 시드 42/43/44 덤프 (베이스의 같은 시드와 짝짓는다)
BASE = ['results/val_samples.jsonl', 'results/val_samples_s43.jsonl', 'results/val_samples_s44.jsonl']
FAMILIES = {
    'teacher': ['results/val_samples_teacher_full_s42.jsonl',
                'results/val_samples_teacher_full_s43.jsonl',
                'results/val_samples_teacher_full_s44.jsonl'],
    'rftmask': ['results/val_samples_rftmask_s42.jsonl',
                'results/val_samples_rftmask_s43.jsonl',
                'results/val_samples_rftmask_s44.jsonl'],
    'dpo':     ['results/val_samples_dpo.jsonl',
                'results/val_samples_dpo_s43.jsonl',
                'results/val_samples_dpo_s44.jsonl'],
    'mt3072':  ['results/val_samples_mt3072_s42.jsonl',
                'results/val_samples_mt3072_s43.jsonl',
                'results/val_samples_mt3072_s44.jsonl'],
    'masked':  ['results/val_samples_masked_s42.jsonl',
                'results/val_samples_masked_s43.jsonl',
                'results/val_samples_masked_s44.jsonl'],
}

# 문제 유형 특징 (전부 텍스트만 본다 — 추론 시에도 로컬 계산 가능해야 하므로)
PATTERNS = [
    ('mod/나머지',    r'\bmod\b|modulo|remainder|divisible|divisor'),
    ('소수/인수',     r'\bprime\b|factoriz|\bfactors?\b|\bgcd\b|\blcm\b|greatest common|least common'),
    ('조합/경우의수',  r'how many ways|number of ways|permutation|combination|\bchoose\b|arrangement|subsets'),
    ('확률/기댓값',   r'probability|expected value|at random|\bdice\b|\bcoin\b|randomly'),
    ('수열/급수',     r'sequence|series|progression|\ba_\{?n|\bs_\{?n|recursive|recurrence'),
    ('기하',         r'triangle|circle|square|rectangle|polygon|angle|radius|diameter|perimeter|\barea\b|volume|hypotenuse|vertex|vertices'),
    ('방정식/다항식',  r'polynomial|equation|solve for|roots of|quadratic|inequality'),
    ('함수 f(x)',    r'\bf\s*\(\s*x|\bg\s*\(\s*x|function|domain of|range of'),
    ('제곱/루트',     r'square root|sqrt|\^2|\^3|squared|cubed|\bpower of'),
    ('floor/ceil',   r'floor|ceiling|greatest integer|least integer|rounded down|rounded up'),
    ('분수',         r'\\frac|\d+\s*/\s*\d+'),
    ('소수점',       r'\d+\.\d+'),
    ('경우분석',      r'at least|at most|either|otherwise|\bcases\b|if and only if'),
]


def feats(q):
    t = (q or '').lower()
    f = {name: bool(re.search(pat, t)) for name, pat in PATTERNS}
    nums = re.findall(r'\d+', q or '')
    f['숫자 8개+'] = len(nums) >= 8
    f['큰수(4자리+)'] = any(len(x) >= 4 for x in nums)
    f['긴 문제(180자+)'] = len(q or '') >= 180
    return f


def vote(samples, n=N, scale=2.0):
    v = [(s['ans'], s['logp']) for s in samples[:n] if s.get('ans') is not None]
    if not v:
        return None
    m = max(lp for _, lp in v)
    w = defaultdict(float)
    for a, lp in v:
        w[a] += math.exp((lp - m) * scale)
    return max(w, key=w.get)


def load(path):
    """id -> 정답여부(0/1)"""
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            out[r['id']] = int(vote(r['samples']) == r['gold'])
    return out


def main():
    with open(TRAIN, encoding='utf-8') as f:
        qmap = {r['id']: r['question'] for r in csv.DictReader(f)}

    base = [load(p) for p in BASE]
    if any(b is None for b in base):
        sys.exit('베이스 덤프 없음')
    ids = sorted(set(base[0]) & set(base[1]) & set(base[2]))
    F = {i: feats(qmap.get(i, '')) for i in ids}
    names = list(next(iter(F.values())).keys())

    print('검증 %d문항 · n=%d · 가중투표 · 베이스 시드42/43/44와 대응' % (len(ids), N))
    print('베이스 정확도: ' + ' / '.join(str(sum(b[i] for i in ids)) for b in base)
          + '  (평균 %.1f)' % (sum(sum(b[i] for i in ids) for b in base) / 3))
    print()
    print('유형별 문항 수:')
    for nm in names:
        k = sum(1 for i in ids if F[i][nm])
        print('  %-16s %4d  (%5.1f%%)' % (nm, k, 100 * k / len(ids)))
    print()

    for fam, paths in FAMILIES.items():
        spec = [load(p) for p in paths]
        if any(s is None for s in spec):
            print('[%s] 덤프 누락 — 건너뜀' % fam)
            continue
        ok = [i for i in ids if all(i in s for s in spec)]
        acc = [sum(s[i] for i in ok) for s in spec]
        bacc = [sum(b[i] for i in ok) for b in base]
        print('=' * 78)
        print('[%s]  자체 %s (평균 %.1f)   vs 베이스 %s (평균 %.1f)  문항 %d'
              % (fam, acc, sum(acc) / 3, bacc, sum(bacc) / 3, len(ok)))

        # 시드별 효용 u_k(x) = spec_k(x) - base_k(x)  in {-1,0,+1}
        U = [{i: spec[k][i] - base[k][i] for i in ok} for k in range(3)]
        tot = [sum(u.values()) for u in U]
        resc = [sum(1 for i in ok if U[k][i] > 0) for k in range(3)]
        harm = [sum(1 for i in ok if U[k][i] < 0) for k in range(3)]
        print('  전체:  구제 %s  훼손 %s  순 %s' % (resc, harm, tot))
        print()
        print('  %-16s%5s  %13s%13s%12s%9s'
              % ('유형', '문항', '구제(3시드)', '훼손(3시드)', '순효용/문항', '부호일치'))
        print('  ' + '-' * 74)

        rows = []
        for nm in names:
            sub = [i for i in ok if F[i][nm]]
            if len(sub) < 15:
                continue
            r = [sum(1 for i in sub if U[k][i] > 0) for k in range(3)]
            h = [sum(1 for i in sub if U[k][i] < 0) for k in range(3)]
            net = [sum(U[k][i] for i in sub) for k in range(3)]
            per = sum(net) / 3 / len(sub)
            rest = [i for i in ok if not F[i][nm]]
            diffs = [sum(U[k][i] for i in sub) / len(sub) - sum(U[k][i] for i in rest) / len(rest)
                     for k in range(3)]
            agree = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
            rows.append((nm, len(sub), per, sum(diffs) / 3, agree))
            print('  %-16s%5d  %13s%13s%+12.3f%9s'
                  % (nm, len(sub), str(r), str(h), per, 'O' if agree else '.'))

        # 순열검정 — "가장 극단적인 칸"이 우연으로 나올 확률
        obs = max(abs(d) for _, _, _, d, _ in rows) if rows else 0.0
        cnt = 0
        TRIALS = 2000
        pool = list(ok)
        for _ in range(TRIALS):
            perm = pool[:]
            random.shuffle(perm)
            mapping = dict(zip(pool, perm))
            best = 0.0
            for nm in names:
                sub = [mapping[i] for i in ok if F[i][nm]]
                if len(sub) < 15:
                    continue
                ss = set(sub)
                rest = [i for i in ok if i not in ss]
                d = sum(sum(U[k][i] for i in sub) / len(sub)
                        - sum(U[k][i] for i in rest) / len(rest) for k in range(3)) / 3
                best = max(best, abs(d))
            if best >= obs:
                cnt += 1
        print('  -> 최대 |차이| = %.3f (문항당), 순열검정 p = %.3f (%d/%d)'
              % (obs, cnt / TRIALS, cnt, TRIALS))
        agreed = [r for r in rows if r[4] and abs(r[3]) >= 0.03]
        print('  -> 3시드 부호일치 & |차이| >= 0.03 인 유형: %s'
              % ([r[0] for r in agreed] if agreed else '없음'))
        print()

    print('=' * 78)
    print('판정 기준 (사전 등록):')
    print('  · 순열검정 p > 0.10 이고 부호 일치 유형이 없으면 -> 전문화 없음, 라우터 불필요')
    print('  · p <= 0.10 이면서 3시드 부호가 일치하는 유형이 있으면 -> learned router로 진행')
    print('    (그 경우에도 최종 판정은 nested-CV 배포 순이득으로 한다)')


if __name__ == '__main__':
    main()
