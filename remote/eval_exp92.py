"""exp92 평가 — 어댑터 스케일 보간 4점 곡선 (α = 0, 0.25, 0.5, 1.0). GPU 0.

사전등록 게이트: 어느 α에서든 가중투표 > 베이스(시드42: 366)이면 시드 43/44 확장.
지표: 가중투표(원자·교정자 모두), pass@32, 고유답 수 — exp69 교훈(표차만 변하고 argmax
불변)을 피하려 분포 지표까지 본다.

사용: PYTHONIOENCODING=utf-8 python remote/eval_exp92.py
"""
import json
import math
import os
from collections import defaultdict

FILES = [
    ('α=0 (베이스)', 'results/val_samples.jsonl'),
    ('α=0.25', 'results/val_samples_ts_a025_s42.jsonl'),
    ('α=0.5', 'results/val_samples_ts_a05_s42.jsonl'),
    ('α=1 (teacher)', 'results/val_samples_teacher_full_s42.jsonl'),
]


def wv(ss, n=32, scale=2.0):
    v = [(s['ans'], s['logp']) for s in ss[:n] if s.get('ans') is not None]
    if not v:
        return None
    m = max(lp for _, lp in v)
    w = defaultdict(float)
    for a, lp in v:
        w[a] += math.exp((lp - m) * scale)
    return max(w, key=w.get)


def main():
    gold = {}
    for l in open('results/val_samples.jsonl', encoding='utf-8'):
        d = json.loads(l)
        gold[d['id']] = d['gold']
    key = json.load(open('experiments/exp88_fable_key.json', encoding='utf-8'))
    corr = {}
    for i, g in gold.items():
        if i in key:
            t = key[i]['truth']
            corr[i] = t if t is not None else ('__X__',)
        else:
            corr[i] = g

    print('exp92 — 보간 곡선 (시드 42, 483문항)')
    print('%-16s %6s %8s %8s %8s %8s' % ('α', '원자', '교정자', 'pass@32', '고유답/문항', '잘림%'))
    print('-' * 62)
    base_dirty = None
    for name, path in FILES:
        if not os.path.exists(path):
            print('%-16s (덤프 없음: %s)' % (name, path))
            continue
        rows = [json.loads(l) for l in open(path, encoding='utf-8')]
        d = c = p = 0
        uniq = 0.0
        tr = tot = 0
        for r in rows:
            ss = r['samples'][:32]
            a = wv(ss)
            g = gold[r['id']]
            d += (a == g)
            t = corr[r['id']]
            c += (not isinstance(t, tuple)) and a == t
            ans = [s['ans'] for s in ss if s.get('ans') is not None]
            p += (g in ans)
            uniq += len(set(ans))
            tr += sum(1 for s in ss if s.get('trunc'))
            tot += len(ss)
        n = len(rows)
        if base_dirty is None:
            base_dirty = d
        print('%-16s %6d %8d %8d %10.2f %8.2f   (원자 베이스 대비 %+d)'
              % (name, d, c, p, uniq / n, 100 * tr / max(1, tot), d - base_dirty))
    print()
    print('게이트: 어느 α에서든 원자 가중투표 > %d 이면 시드 43/44 확장' % base_dirty)


if __name__ == '__main__':
    main()
