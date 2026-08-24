# -*- coding: utf-8 -*-
"""exp93 채점 — full FT 체크포인트별 val 덤프를 교정 정답지(exp88)로 채점.

사전등록(prereg_exp93_bigft.md) 게이트:
  G1: 어느 ckpt든 교정자 가중투표 >= 383 → 최고 ckpt에 시드 43/44 확장
  G2: 3시드 평균 >= 384 AND 최저 >= 378 → 사용자 승인 요청 (자동 채택 금지)
베이스라인: 교정자 380 / 원자 366 (n=32, seed 42).

사용: PYTHONIOENCODING=utf-8 uv run python remote/eval_exp93.py
→ 표 출력 + results/eval_exp93.json 기록 (작은 파일 — 커밋 대상)
"""
import glob
import json
import math
import os
import re
from collections import defaultdict

BASE_DIRTY, BASE_CORR = 366, 380
G1 = 383


def wvote(samps, scale=2.0):
    v = [(s['ans'], s['logp']) for s in samps if s.get('ans') is not None]
    if not v:
        return None
    m = max(lp for _, lp in v)
    w = defaultdict(float)
    for a, lp in v:
        w[a] += math.exp((lp - m) * scale)
    return max(w, key=w.get)


def score(path, gold, corr):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    d = c = p = tr = tot = 0
    uniq = 0.0
    for r in rows:
        ss = r['samples'][:32]
        a = wvote(ss)
        d += (a == gold[r['id']])
        t = corr[r['id']]
        if t is not None:
            c += (a == t)
            ans = [s['ans'] for s in ss if s.get('ans') is not None]
            p += (t in ans)
        ans = [s['ans'] for s in ss if s.get('ans') is not None]
        uniq += len(set(ans))
        tr += sum(1 for s in ss if s.get('trunc'))
        tot += len(ss)
    n = len(rows)
    return dict(n=n, dirty=d, corr=c, passn=p, uniq=round(uniq / n, 2),
                trunc_pct=round(100 * tr / max(1, tot), 2))


def main():
    gold = {}
    for l in open('results/val_samples.jsonl', encoding='utf-8'):
        r = json.loads(l)
        gold[r['id']] = r['gold']
    key = json.load(open('experiments/exp88_fable_key.json', encoding='utf-8'))
    corr = {i: (key[i]['truth'] if i in key else g) for i, g in gold.items()}

    files = sorted(glob.glob('results/val_samples_e93_c*_s4*.jsonl'),
                   key=lambda f: (int(re.search(r'_c(\d+)_', f).group(1)), f))
    if not files:
        print('e93 덤프 없음')
        return

    out = {}
    print('%-34s %6s %8s %8s %8s %7s' % ('덤프', '원자', '교정자', 'pass@32', '고유답', '잘림%'))
    print('-' * 76)
    for f in files:
        s = score(f, gold, corr)
        out[os.path.basename(f)] = s
        print('%-34s %6d %8d %8d %8.2f %7.2f' % (os.path.basename(f),
              s['dirty'], s['corr'], s['passn'], s['uniq'], s['trunc_pct']))
    print()
    print(f'베이스라인: 원자 {BASE_DIRTY} / 교정자 {BASE_CORR} (n=32 s42)')
    best = max((v['corr'], k) for k, v in out.items() if '_s42' in k)
    verdict = 'G1 통과 (>=383) → 시드 43/44 확장' if best[0] >= G1 else 'G1 기각 (<383) → 실험 종료, 동결 유지'
    print(f'최고 ckpt(s42): {best[1]} 교정자 {best[0]} → {verdict}')
    json.dump({'results': out, 'baseline': {'dirty': BASE_DIRTY, 'corr': BASE_CORR},
               'best_s42': {'file': best[1], 'corr': best[0]}, 'verdict': verdict},
              open('results/eval_exp93.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('저장: results/eval_exp93.json')


if __name__ == '__main__':
    main()
