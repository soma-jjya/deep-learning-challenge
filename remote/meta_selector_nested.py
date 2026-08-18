"""exp55c — nested CV로 "실제 배포했다면 몇 문항을 벌었나"를 정직하게 잰다. GPU 불필요.

## 왜 필요한가
exp55b의 `max OOF net gain`은 **여러 k를 다 본 뒤 가장 좋은 k를 고른** 값이다.
실전에서는 그 k를 미리 알 수 없으므로 그 수치는 상한에 가깝다.

여기서는 결정 규칙(임계값)을 **바깥 fold를 보지 않고** 안쪽 CV에서만 고른 뒤,
한 번도 보지 않은 fold에 그대로 적용한다:

    outer 5-fold
      ├─ train 4 folds → inner CV로 OOF 점수 → 최적 임계값 선택
      └─ held-out 1 fold → 그 임계값 그대로 적용해 net gain 집계

임계값(k가 아니라 score threshold)을 고르는 이유: fold 크기가 달라 k는 이전되지 않는다.

사용: python remote/meta_selector_nested.py
"""
import argparse
import random
import sys

sys.path.insert(0, 'remote')
from meta_selector import standardize, fit_logreg, predict
from meta_selector_utility import build, SMALL


def fit_two_heads(data, feats, tr, keys):
    tr = [i for i in tr if data[i]['u'] != 0]        # neutral은 학습에서 제외
    Xtr = [feats[i] for i in tr]
    wa = fit_logreg(Xtr, [1 if data[i]['u'] == +1 else 0 for i in tr], keys)
    wb = fit_logreg(Xtr, [1 if data[i]['u'] == -1 else 0 for i in tr], keys)
    return wa, wb


def score_of(wa, wb, feats, i, keys):
    return predict(wa, feats[i], keys) - predict(wb, feats[i], keys)


def pick_threshold(data, feats, tr, keys, inner_seed, inner_folds=4):
    """안쪽 CV로만 임계값을 고른다 — 바깥 fold는 절대 보지 않는다."""
    idx = list(tr)
    random.Random(inner_seed).shuffle(idx)
    parts = [idx[i::inner_folds] for i in range(inner_folds)]
    sc = {}
    for k in range(inner_folds):
        te = set(parts[k])
        sub = [i for i in idx if i not in te]
        wa, wb = fit_two_heads(data, feats, sub, keys)
        for i in parts[k]:
            sc[i] = score_of(wa, wb, feats, i, keys)
    best_thr, best_net = 1.0, 0          # 기본값 = 아무것도 뒤집지 않음(net 0)
    cands = sorted({round(v, 3) for v in sc.values()}, reverse=True)
    for thr in cands:
        net = sum(data[i]['u'] for i in idx if sc[i] >= thr)
        if net > best_net:
            best_net, best_thr = net, thr
    return best_thr


def nested_eval(data, keys, outer_seed, outer_folds=5):
    N = len(data)
    feats = standardize([d['f'] for d in data], keys)
    idx = list(range(N))
    random.Random(outer_seed).shuffle(idx)
    parts = [idx[i::outer_folds] for i in range(outer_folds)]
    total = res = harm = flips = 0
    for k in range(outer_folds):
        te = parts[k]
        tr = [i for i in idx if i not in set(te)]
        thr = pick_threshold(data, feats, tr, keys, inner_seed=outer_seed * 7 + k)
        wa, wb = fit_two_heads(data, feats, tr, keys)
        for i in te:
            if score_of(wa, wb, feats, i, keys) >= thr:
                flips += 1
                total += data[i]['u']
                if data[i]['u'] == +1:
                    res += 1
                elif data[i]['u'] == -1:
                    harm += 1
    return total, flips, res, harm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=8)
    args = ap.parse_args()

    data = build(32)
    nres = sum(1 for d in data if d['u'] == +1)
    nharm = sum(1 for d in data if d['u'] == -1)
    print(f'접전 {len(data)}문항 — rescue {nres} / harm {nharm} / '
          f'neutral {len(data)-nres-nharm}')
    print('nested CV: 임계값을 안쪽에서만 고르고 한 번도 안 본 fold에 적용')
    print()

    allkeys = sorted(data[0]['f'].keys())
    for name, keys in (('전체 18개 특징', allkeys), ('축소 6개 특징', SMALL)):
        rows = [nested_eval(data, keys, 200 + r) for r in range(args.reps)]
        nets = [r[0] for r in rows]
        m = sum(nets) / len(nets)
        print(f'── {name} ──')
        print(f'   시드별 배포 net gain: {nets}')
        print(f'   평균 {m:+.2f}문항 / 최대 {max(nets):+d} / 최소 {min(nets):+d} / '
              f'양수 시드 {sum(1 for n in nets if n > 0)}/{len(nets)}')
        f = sum(r[1] for r in rows) / len(rows)
        rr = sum(r[2] for r in rows) / len(rows)
        hh = sum(r[3] for r in rows) / len(rows)
        print(f'   평균 뒤집기 {f:.1f}개 (rescue {rr:.1f} / harm {hh:.1f})')
        print()

    print('해석: 이 값이 배포 시 기대치에 가장 가깝다. exp55b의 max OOF net gain')
    print('      (평균 +1.8)은 k를 사후 최적화한 상한이었다.')


if __name__ == '__main__':
    main()
