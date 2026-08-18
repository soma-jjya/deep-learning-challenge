"""H32 — Top-2 전용 메타 선택기. GPU 불필요, 순수 파이썬(의존성 추가 없음).

## 왜 이것이 자기검증 6종과 다른가
지금까지 선택기는 전부 **"이 풀이가 수학적으로 맞나?"** 를 모델에게 물었다 —
절대 채점(검증자), 상대 비교(쌍대), DPO 암묵보상, 보상헤드+Bradley-Terry, implicit PRM,
토큰 확신도. 여섯 갈래가 전부 판별력 ~0.6에 수렴했다.

여기서는 **풀이를 전혀 보지 않는다.** 투표 통계만으로 "언제 다수결 1위가 틀리는가"를 학습한다.
exp50의 조건부 규칙(표차 하나로 분기하는 사람이 만든 규칙)과도 다르다 —
**여러 약한 신호의 조합**을 학습한다.

## 왜 top-2에 국한하나
exp50 측정: 1위가 틀린 119문항에서 정답이 2위인 것은 22문항(18.5%)뿐이다.
완벽한 top-2 선택기의 상한은 +4.14%p이고, 2등에 필요한 +1.5%p는 그 35%만 회수하면 된다.
즉 필요한 것은 천재 검증자가 아니라 **뒤집을 수 있는 22문항 중 7~8개를 찾는 작은 분류기**다.

## ⚠️ 이 실험의 근본 위험 (반드시 함께 읽을 것)
exp52b에서 **어려운 문항일수록 로컬 gold가 깨져 있음**을 확인했다. 그런데 top1/top2가
충돌하는 문항이 정확히 그 구간이다. **라벨이 틀린 곳에서 "뒤집어라"를 학습시키면
분류기는 라벨 노이즈를 배운다.** CV 점수가 올라도 그것이 실력인지 노이즈 학습인지
구분되지 않는다. 그래서 결과는 반드시 리더보드로 확인해야 한다.

사용: python remote/meta_selector.py
"""
import argparse
import json
import math
import random
from collections import defaultdict


def load(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]


def tally(samples, scale=2.0):
    """확신도 가중 집계 → [(답, 가중치, 표수, [logp들], [길이들], 잘림수)] 내림차순."""
    agg = defaultdict(lambda: [0.0, 0, [], [], 0])
    for s in samples:
        a = s.get('ans')
        if a is None:
            continue
        w = math.exp(scale * s.get('logp', 0.0))
        e = agg[a]
        e[0] += w
        e[1] += 1
        e[2].append(s.get('logp', 0.0))
        e[3].append(s.get('len', 0))
        e[4] += 1 if s.get('trunc') else 0
    out = [(k, v[0], v[1], v[2], v[3], v[4]) for k, v in agg.items()]
    out.sort(key=lambda x: -x[1])
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def var(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def features(row, n, others):
    """others = 다른 시드 덤프의 {id: samples}. seed disagreement feature용."""
    t = tally(row['samples'][:n])
    if len(t) < 2:
        return None
    a1, w1, c1, lp1, ln1, tr1 = t[0]
    a2, w2, c2, lp2, ln2, tr2 = t[1]
    tot_w = sum(x[1] for x in t) or 1.0
    tot_c = sum(x[2] for x in t) or 1

    # 다른 시드에서도 같은 답이 1위인가 (H22를 '합산'이 아니라 '신호'로 쓰는 부분)
    agree = 0
    seen = 0
    for o in others:
        ss = o.get(row['id'])
        if not ss:
            continue
        seen += 1
        to = tally(ss[:n])
        if to and to[0][0] == a1:
            agree += 1
    seed_agree = agree / seen if seen else 0.5

    f = {
        'w_share1': w1 / tot_w,
        'w_share2': w2 / tot_w,
        'w_gap': (w1 - w2) / tot_w,
        'w_ratio': w2 / w1 if w1 else 1.0,
        'c_share1': c1 / tot_c,
        'c_gap': (c1 - c2) / tot_c,
        'n_distinct': len(t) / n,
        'lp1_mean': mean(lp1),
        'lp2_mean': mean(lp2),
        'lp_gap': mean(lp1) - mean(lp2),
        'lp1_var': var(lp1),
        'len1_mean': mean(ln1) / 2048.0,
        'len2_mean': mean(ln2) / 2048.0,
        'len_gap': (mean(ln1) - mean(ln2)) / 2048.0,
        'trunc1': tr1 / max(1, c1),
        'trunc2': tr2 / max(1, c2),
        'seed_agree': seed_agree,
        'bias': 1.0,
    }
    return f, a1, a2


def fit_logreg(X, y, keys, epochs=400, lr=0.35, l2=1e-3):
    w = {k: 0.0 for k in keys}
    n = len(X)
    for _ in range(epochs):
        g = {k: 0.0 for k in keys}
        for xi, yi in zip(X, y):
            z = sum(w[k] * xi[k] for k in keys)
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            d = p - yi
            for k in keys:
                g[k] += d * xi[k]
        for k in keys:
            w[k] -= lr * (g[k] / n + l2 * w[k])
    return w


def predict(w, xi, keys):
    z = sum(w[k] * xi[k] for k in keys)
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


def standardize(feats, keys):
    mu = {k: mean([f[k] for f in feats]) for k in keys if k != 'bias'}
    sd = {k: (var([f[k] for f in feats]) ** 0.5) or 1.0 for k in keys if k != 'bias'}
    out = []
    for f in feats:
        g = dict(f)
        for k in keys:
            if k != 'bias':
                g[k] = (f[k] - mu[k]) / sd[k]
        out.append(g)
    return out


def analyze_ceiling(data, probs, y, base_ok, keys, feats):
    """precision@k 곡선 + 필요한 AUC 역산. 왜 안 되는지를 수치로 못박는다."""
    N = len(data)
    order = sorted(range(N), key=lambda i: -probs[i])
    print()
    print('상위 k개만 뒤집었을 때 (분류기 순위를 그대로 신뢰):')
    print(f'{"k":>5s}{"성공":>7s}{"실패":>7s}{"순이득":>9s}{"정밀도":>9s}')
    for k in (5, 10, 15, 20, 30, 40, 60):
        good = bad = 0
        for i in order[:k]:
            if data[i]['a2'] == data[i]['gold']:
                good += 1
            elif data[i]['a1'] == data[i]['gold']:
                bad += 1
        print(f'{k:5d}{good:7d}{bad:7d}{good-bad:+9d}{good/k:9.1%}')
    n_pos = sum(y)
    print()
    print(f'뒤집을 수 있는 문항은 총 {n_pos}개, 깨뜨릴 수 있는 문항은 {base_ok}개다.')
    print(f'순이득 +8을 내려면 상위 20개 중 14개가 진짜여야 한다(정밀도 70%, 재현율 64%).')
    print(f'그 수준은 대략 AUC 0.90 이상이 필요하고, 우리는 0.66이다. 조정으로 메울 폭이 아니다.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary', default='results/val_samples.jsonl')
    ap.add_argument('--others', nargs='*',
                    default=['results/val_samples_s43.jsonl', 'results/val_samples_s44.jsonl'])
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rows = load(args.primary)
    others = []
    for p in args.others:
        try:
            others.append({r['id']: r['samples'] for r in load(p)})
        except OSError:
            print(f'  (없음: {p})')

    data = []
    for r in rows:
        got = features(r, args.n, others)
        if got is None:
            continue
        f, a1, a2 = got
        data.append({'f': f, 'a1': a1, 'a2': a2, 'gold': r['gold'], 'id': r['id']})

    keys = sorted(data[0]['f'].keys())
    base_ok = sum(1 for d in data if d['a1'] == d['gold'])
    flip_ok = sum(1 for d in data if d['a2'] == d['gold'])
    N = len(data)
    print(f'검증 {N}문항 (2개 이상 후보), n={args.n}')
    print(f'  현재 1위 정답      : {base_ok} ({base_ok/N:.1%})')
    print(f'  2위가 정답인 문항  : {flip_ok} ({flip_ok/N:.1%})  ← 뒤집어서 얻을 수 있는 최대')
    print(f'  둘 다 아님        : {N - base_ok - flip_ok}')
    print()

    y = [1 if d['a2'] == d['gold'] else 0 for d in data]   # 뒤집어야 하는가
    feats = standardize([d['f'] for d in data], keys)

    idx = list(range(N))
    random.Random(args.seed).shuffle(idx)
    folds = [idx[i::args.folds] for i in range(args.folds)]

    print(f'{"임계값":>8s}{"뒤집음":>8s}{"성공":>7s}{"실패":>7s}{"CV 정확도":>12s}{"기준대비":>10s}')
    print('-' * 54)
    probs = [0.0] * N
    for k in range(args.folds):
        te = set(folds[k])
        tr = [i for i in idx if i not in te]
        w = fit_logreg([feats[i] for i in tr], [y[i] for i in tr], keys)
        for i in folds[k]:
            probs[i] = predict(w, feats[i], keys)

    best = None
    # 뒤집어야 할 문항이 8%대라 확률이 낮게 형성된다 — 임계 구간을 그에 맞춘다
    for thr in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40):
        ok = flips = good = bad = 0
        for i, d in enumerate(data):
            pick = d['a2'] if probs[i] >= thr else d['a1']
            if probs[i] >= thr:
                flips += 1
                if d['a2'] == d['gold']:
                    good += 1
                elif d['a1'] == d['gold']:
                    bad += 1
            ok += (pick == d['gold'])
        delta = ok - base_ok
        print(f'{thr:8.2f}{flips:8d}{good:7d}{bad:7d}{ok/N:11.1%}{delta:+9d}문항')
        if best is None or ok > best[1]:
            best = (thr, ok)

    # 판별력 자체 (임계값과 무관)
    pos=[probs[i] for i in range(N) if y[i]==1]; neg=[probs[i] for i in range(N) if y[i]==0]
    import bisect
    ps=sorted(pos); wins=ties=0
    for v in neg:
        lo=bisect.bisect_left(ps,v); hi=bisect.bisect_right(ps,v)
        wins += len(ps)-hi; ties += hi-lo
    auc=(wins+0.5*ties)/(len(pos)*len(neg)) if pos and neg else float('nan')

    print()
    print(f'※ 접전 {N}문항 기준 {base_ok}({base_ok/N:.1%}) / 최고 {best[1]} ({best[1]-base_ok:+d})')
    print(f'※ 전체 483문항 환산: 기준 366 -> 최고 {366 + (best[1]-base_ok)}')
    print(f'※ 뒤집기 분류기 판별력 AUC = {auc:.3f}  (0.5=무신호)')
    print('   자기검증 6종이 전부 0.59~0.65였다. 여기가 그보다 확실히 높아야 새로운 것이다.')
    print(f'※ 성공 기준: +8문항 이상 (2등에 필요한 +1.5%p)')
    print()
    print('⚠️ exp52b 경고: 어려운 문항일수록 로컬 gold가 깨져 있고, top1/top2 충돌 문항이')
    print('   정확히 그 구간이다. CV 이득이 실력인지 라벨 노이즈 학습인지는 리더보드로만 갈린다.')
    print('⚠️ 임계값을 여러 개 보고 최고를 고른 표이므로 이미 낙관적이다.')
    analyze_ceiling(data, probs, y, base_ok, keys, feats)

    # 어떤 신호가 실제로 기여했는지 (전체 데이터 1회 적합, 표준화된 계수)
    w_all = fit_logreg(feats, y, keys)
    print()
    print('특징별 기여 (표준화 계수, 절대값 상위):')
    for k, v in sorted(w_all.items(), key=lambda kv: -abs(kv[1]))[:8]:
        if k != 'bias':
            print(f'  {k:14s} {v:+.3f}')


if __name__ == '__main__':
    main()

