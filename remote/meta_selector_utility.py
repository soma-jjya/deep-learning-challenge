"""exp55b — 비용 민감(cost-sensitive) 메타 선택기. GPU 불필요, 의존성 추가 없음.

## exp55(meta_selector.py)의 오류를 고친다
exp55는 "뒤집어야 하는가"를 이진 분류로 놓고, 비-rescue를 전부 harm으로 취급해
"상위 20개 중 14개가 진짜여야 +8"이라고 계산했다. **틀렸다.**
실제로 접전 267문항은 세 부류다:

  rescue  : top1 오답 & top2 정답  →  뒤집으면 +1   (22개)
  harm    : top1 정답              →  뒤집으면 −1   (153개)
  neutral : 둘 다 오답             →  뒤집어도  0   (92개, 34%)

neutral이 3분의 1이므로 `8 rescue / 0 harm / 12 neutral`도 +8이다.
최적화 대상은 P(flip)이 아니라 **기대 이득 = P(rescue) − P(harm)** 이고,
평가 지표는 AUC가 아니라 **max OOF net gain = max_k [rescued(k) − harmed(k)]** 다.

## exp55가 남긴 진짜 신호 (닫으면 안 되는 이유)
모집단 rescue:harm = 22:153 = 1:6.95인데, 상위 k에서는 1:2.0 ~ 1:2.75로 올라왔다.
**harm 대비 rescue가 2.5~3.5배 농축**된 것이다. 아직 rescue > harm 경계는 못 넘었지만
"최상위에도 신호가 없다"는 서술은 부정확했다.

## 과적합 대비
positive가 22개뿐인데 특징이 18개면 과적합한다. 축소 특징셋(6개)과 전체를 함께 재고,
CV 시드를 여러 개 돌려 안정성을 본다.

사용: python remote/meta_selector_utility.py
"""
import argparse
import math
import random
import sys

sys.path.insert(0, 'remote')
from meta_selector import load, features, standardize, fit_logreg, predict

SMALL = ['w_gap', 'w_ratio', 'c_gap', 'n_distinct', 'lp_gap', 'bias']


def build(n=32):
    rows = load('results/val_samples.jsonl')
    others = []
    for p in ('results/val_samples_s43.jsonl', 'results/val_samples_s44.jsonl'):
        try:
            others.append({r['id']: r['samples'] for r in load(p)})
        except OSError:
            pass
    data = []
    for r in rows:
        g = features(r, n, others)
        if not g:
            continue
        f, a1, a2 = g
        if a1 == r['gold']:
            u = -1          # harm: 뒤집으면 손해
        elif a2 == r['gold']:
            u = +1          # rescue: 뒤집으면 이득
        else:
            u = 0           # neutral
        data.append({'f': f, 'u': u})
    return data


def oof_scores(data, keys, cv_seed, folds=5, drop_neutral=True):
    """2-head: P(rescue)와 P(harm)을 따로 학습해 score = p_res - p_harm."""
    N = len(data)
    feats = standardize([d['f'] for d in data], keys)
    idx = list(range(N))
    random.Random(cv_seed).shuffle(idx)
    parts = [idx[i::folds] for i in range(folds)]
    score = [0.0] * N
    for k in range(folds):
        te = set(parts[k])
        tr = [i for i in idx if i not in te]
        if drop_neutral:
            tr = [i for i in tr if data[i]['u'] != 0]   # neutral은 학습에서 제외
        y_res = [1 if data[i]['u'] == +1 else 0 for i in tr]
        y_harm = [1 if data[i]['u'] == -1 else 0 for i in tr]
        Xtr = [feats[i] for i in tr]
        wa = fit_logreg(Xtr, y_res, keys)
        wb = fit_logreg(Xtr, y_harm, keys)
        for i in parts[k]:
            score[i] = predict(wa, feats[i], keys) - predict(wb, feats[i], keys)
    return score


def net_gain_curve(data, score):
    order = sorted(range(len(data)), key=lambda i: -score[i])
    res = harm = 0
    best = (0, 0, 0, 0)   # (net, k, rescued, harmed)
    curve = []
    for k, i in enumerate(order, 1):
        u = data[i]['u']
        if u == +1:
            res += 1
        elif u == -1:
            harm += 1
        net = res - harm
        curve.append((k, res, harm, net))
        if net > best[0]:
            best = (net, k, res, harm)
    return curve, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--reps', type=int, default=8, help='CV 시드 반복 수')
    args = ap.parse_args()

    data = build(args.n)
    nres = sum(1 for d in data if d['u'] == +1)
    nharm = sum(1 for d in data if d['u'] == -1)
    nneu = sum(1 for d in data if d['u'] == 0)
    print(f'접전 {len(data)}문항 — rescue {nres} / harm {nharm} / neutral {nneu}')
    print(f'  모집단 rescue:harm = 1:{nharm/nres:.2f}')
    print(f'  성공 기준: max OOF net gain >= +8  (이론 상한 +{nres})')
    print()

    allkeys = sorted(data[0]['f'].keys())
    for name, keys in (('전체 18개 특징', allkeys), ('축소 6개 특징', SMALL)):
        bests = []
        for rep in range(args.reps):
            sc = oof_scores(data, keys, cv_seed=100 + rep)
            _, best = net_gain_curve(data, sc)
            bests.append(best)
        nets = [b[0] for b in bests]
        m = sum(nets) / len(nets)
        print(f'── {name} (CV 시드 {args.reps}회) ──')
        print(f'   max OOF net gain: {nets}')
        print(f'   평균 {m:+.1f}문항 / 최대 {max(nets):+d} / 최소 {min(nets):+d}')
        # 대표 곡선 한 개
        sc = oof_scores(data, keys, cv_seed=100)
        curve, best = net_gain_curve(data, sc)
        print(f'   대표 실행: 상위 {best[1]}개 뒤집을 때 rescue {best[2]} / harm {best[3]} = {best[0]:+d}')
        pts = [c for c in curve if c[0] in (5, 10, 20, 30, 50, 80)]
        print('   k / rescue / harm / net : ' +
              '  '.join(f'{k}:{r}/{h}/{n:+d}' for k, r, h, n in pts))
        print()

    print('판정: max OOF net gain이 8 미만이고 CV 시드 간 부호가 흔들리면 이 축을 닫는다.')
    print('참고: 이 값은 k를 사후에 최적화한 것이라 이미 낙관적이다. 실전에서는 k를')
    print('      미리 정해야 하므로 실제 이득은 이보다 작다.')


if __name__ == '__main__':
    main()
