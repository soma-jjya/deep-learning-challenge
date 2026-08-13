"""분야별 정확도 분해 (H26 사전 진단) — 분야 특화가 말이 되는지부터 확인. GPU 불필요.

"문제를 분야별로 나눠 각각 특화 어댑터를 태우면 오르지 않을까"라는 아이디어가 성립하려면
**분야별 정확도가 실제로 크게 달라야** 한다. 전 분야가 고르게 75%면 특화할 대상이 없다.
게다가 실패 문항이 특정 분야에 몰려 있는지, 아니면 전 분야에 흩어져 있는지에 따라
교사 증류의 표적도 달라진다. 그래서 학습을 하기 전에 이 분해부터 한다.

분류는 키워드 휴리스틱이다. 완벽하지 않지만 **분야 간 정확도 차이가 존재하는가**라는
1차 질문에는 충분하고, 추론 시에도 로컬에서 돌릴 수 있어야 하므로(외부 API 금지)
어차피 이 수준의 분류기를 쓰게 된다.

사용: PYTHONIOENCODING=utf-8 python remote/analyze_by_category.py
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict

csv.field_size_limit(10 ** 7)

# 순서가 중요하다 — 앞쪽이 더 구체적인 분류. 첫 매칭을 채택한다.
CATEGORIES = [
    ('미적분', r'derivative|integral|\bintegrate\b|limit as|tangent line|f\'\(|dy/dx|maxima|minima|monotonic'),
    ('수열/급수', r'sequence|arithmetic progression|geometric progression|\bseries\b|sum of the first|a_\{?n|S_\{?n|recursive'),
    ('확률/통계', r'probability|expected value|\bmean\b|median|variance|standard deviation|random|dice|coin|drawn at random'),
    ('조합론', r'how many ways|number of ways|arrangement|permutation|combination|\bchoose\b|distinct arrangements|subsets'),
    ('정수론', r'divisor|divisible|\bprime\b|remainder|modulo|\bmod\b|greatest common|least common|gcd|lcm|digits of|base-?\d+ representation'),
    ('기하', r'triangle|circle|square|rectangle|polygon|angle|radius|diameter|perimeter|\barea\b|volume|parallel|perpendicular|vertex|vertices|coordinate plane|hypotenuse'),
    ('대수/방정식', r'polynomial|equation|solve for|roots of|factor|quadratic|inequality|\bmatrix\b|vector|complex number|logarithm|\blog_|exponent'),
    ('함수', r'\bf\(x\)|\bg\(x\)|function|domain|range of'),
]


def classify(q):
    t = (q or '').lower()
    for name, pat in CATEGORIES:
        if re.search(pat, t):
            return name
    return '기타/문장제'


def weighted_vote(samples, n=32, scale=2.0):
    v = [(s['ans'], s['logp']) for s in samples[:n] if s['ans'] is not None]
    if not v:
        return None
    m = max(lp for _, lp in v)
    w = defaultdict(float)
    for a, lp in v:
        w[a] += math.exp((lp - m) * scale)
    return max(w, key=w.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', default=[
        'results/val_samples.jsonl', 'results/val_samples_s43.jsonl', 'results/val_samples_s44.jsonl'])
    ap.add_argument('--train', default='deep-learning-challenge-2026/deep_chal_math_train.csv')
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()

    with open(args.train, encoding='utf-8') as f:
        qmap = {r['id']: r['question'] for r in csv.DictReader(f)}

    sets = [[json.loads(l) for l in open(p, encoding='utf-8')] for p in args.files]

    # 시드별로 따로 세고 평균 — 단일 시드 수치를 믿지 않는다는 원칙(exp48) 적용
    stats = defaultdict(lambda: {'n': 0, 'ok': [0] * len(sets), 'pass': 0, 'nocorrect': 0})
    for si, rows in enumerate(sets):
        for r in rows:
            cat = classify(qmap.get(r['id'], ''))
            s = stats[cat]
            if si == 0:
                s['n'] += 1
                ans = [x['ans'] for x in r['samples'][:args.n] if x['ans'] is not None]
                s['pass'] += (r['gold'] in ans)
                s['nocorrect'] += (r['gold'] not in ans)
            s['ok'][si] += (weighted_vote(r['samples'], args.n) == r['gold'])

    total_n = sum(v['n'] for v in stats.values())
    print(f'검증 {total_n}문항, n={args.n}, 시드 {len(sets)}개 평균')
    print()
    print(f'{"분야":14s}{"문항":>6s}{"비중":>7s}{"정확도":>9s}{"pass@32":>10s}{"선택여지":>9s}{"정답부재":>9s}')
    print('-' * 66)

    rows_out = []
    for cat, v in sorted(stats.items(), key=lambda kv: -kv[1]['n']):
        n = v['n']
        acc = sum(v['ok']) / len(sets) / n
        pas = v['pass'] / n
        rows_out.append((cat, n, acc, pas, v['nocorrect']))
        print(f'{cat:14s}{n:>6d}{n/total_n:>7.1%}{acc:>9.1%}{pas:>10.1%}'
              f'{pas-acc:>9.1%}{v["nocorrect"]:>9d}')

    big = [r for r in rows_out if r[1] >= 20]
    accs = [r[2] for r in big]
    print()
    print('분야 간 정확도 편차 (문항 20개 이상 분야만): '
          f'최저 {min(accs):.1%} ~ 최고 {max(accs):.1%}, '
          f'폭 {(max(accs)-min(accs))*100:.1f}%p')
    print('⚠️ 문항 20개 미만 분야(확률/통계·조합론·함수·미적분)는 표본이 작아 오차가 크다 — '
          '방향만 참고할 것')

    # 실패 질량이 어디 몰려 있나 — 교사 증류 표적 선정에 직결
    print()
    print('실패 문항(오답) 분포 — 어디에 표적을 둘 것인가:')
    fails = sorted(((r[0], round(r[1] * (1 - r[2]))) for r in rows_out), key=lambda x: -x[1])
    tot_fail = sum(f[1] for f in fails)
    cum = 0
    for cat, f in fails:
        cum += f
        print(f'  {cat:14s} 오답 {f:>4d}건 ({f/tot_fail:>5.1%})  누적 {cum/tot_fail:>5.1%}')

    print()
    print('해석 기준:')
    print('  · 분야 간 정확도 폭이 10%p 미만이면 → 특화할 대상이 없다(전 분야 고르게 어렵다)')
    print('  · "정답부재"가 큰 분야 = 32번 풀어도 정답이 안 나오는 분야 = 생성 능력 자체가 부족')
    print('    → 이 분야가 교사 증류의 1순위 표적이다')


if __name__ == '__main__':
    main()
