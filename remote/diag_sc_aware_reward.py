"""exp72 — SC-aware marginal-credit 보상의 **오프라인 진단**. GPU 0.

## 무엇을 재는가
GRPO를 6~10시간 돌리기 전에, 새 보상이 **기존 exact-match 보상보다 실제로 더 많은
그래디언트 신호를 만드는지**를 기존 덤프만으로 확인한다.
(exp62·exp69에서 '전제부터 재기'가 며칠을 아낀 전례를 따른다.)

## 두 보상
exact-match (우리 exp12·exp24가 쓴 것):
    r_i = 1 if ans_i == gold else 0

SC-aware marginal credit:
    M(S) = c_gold − max_{a≠gold} c_a          (정답과 최강 오답의 표차)
    r_i  = M(S) − M(S\i)
  → 정답 rollout +1 / 지배적 오답 rollout −1 / 그 외 오답 0

## 왜 다를 수 있는가
GRPO는 그룹 내 보상을 표준화해 advantage를 만든다. 따라서 **그룹 내 보상이 전부 같으면
advantage가 0이 되어 그래디언트가 사라진다.**
- 정답이 그룹에 하나도 없는 경우: exact-match는 전원 0 → **신호 없음**
- 같은 경우 SC-aware는 지배적 오답에 −1 → **신호 있음**
이것이 우리 실패 구간(32표 어디에도 정답 없음 30%)과 정확히 겹친다.

## ⚠️ 이 진단이 답하지 못하는 것
'신호가 있다'와 '학습이 효과가 있다'는 다르다. 특히 정답이 아예 없는 문항에서
지배적 오답만 눌러봐야 올릴 정답이 없다. 이 진단은 **go/no-go 게이트**일 뿐이며,
통과해도 pilot에서 SC·gold 득표율·오답 다양성을 함께 봐야 한다.

사용: python remote/diag_sc_aware_reward.py
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict


def wvote(samples, scale=2.0):
    c = defaultdict(float)
    for s in samples:
        a = s.get('ans')
        if a is not None:
            c[a] += math.exp(scale * s.get('logp', 0.0))
    return max(c, key=c.get) if c else None


def rewards_exact(ans, gold):
    return [1.0 if a == gold else 0.0 for a in ans]


def rewards_sc_aware(ans, gold):
    """r_i = M(S) - M(S\i), M = c_gold - max_{a!=gold} c_a."""
    cnt = Counter(a for a in ans if a is not None)

    def M(c):
        cg = c.get(gold, 0)
        wrong = [v for k, v in c.items() if k != gold]
        return cg - (max(wrong) if wrong else 0)

    base = M(cnt)
    out = []
    for a in ans:
        c2 = Counter(cnt)
        if a is not None:
            c2[a] -= 1
            if c2[a] == 0:
                del c2[a]
        out.append(base - M(c2))
    return [float(x) for x in out]


def stats(rs):
    m = sum(rs) / len(rs)
    v = sum((x - m) ** 2 for x in rs) / len(rs)
    return m, math.sqrt(v)


def advantages(rs):
    m, sd = stats(rs)
    if sd == 0:
        return [0.0] * len(rs)
    return [(x - m) / sd for x in rs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='results/val_samples.jsonl')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--draws', type=int, default=40, help='문항당 그룹 표집 횟수')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples, encoding='utf-8')]
    rng = random.Random(args.seed)

    # 문항 분류: SC가 맞음 / SC 틀렸는데 정답이 표본에 있음 / 정답이 아예 없음
    buckets = {'SC정답': [], 'SC오답·정답있음': [], '정답없음': []}
    for r in rows:
        ss = r['samples'][:args.n]
        gold = r['gold']
        present = gold in {s.get('ans') for s in ss}
        if wvote(ss) == gold:
            buckets['SC정답'].append(r)
        elif present:
            buckets['SC오답·정답있음'].append(r)
        else:
            buckets['정답없음'].append(r)

    print(f'검증 {len(rows)}문항, 그룹 {args.group}개 × 문항당 {args.draws}회 표집')
    for k, v in buckets.items():
        print(f'  {k:16s} {len(v):4d}문항')
    print()

    hdr = (f'{"구간":16s}{"보상":>10s}{"신호있는 그룹":>14s}'
           f'{"정답 advantage":>16s}{"지배오답 adv":>14s}')
    print(hdr)
    print('-' * len(hdr))

    summary = {}
    for bname, brows in buckets.items():
        if not brows:
            continue
        for rname, rfn in (('exact-match', rewards_exact), ('SC-aware', rewards_sc_aware)):
            usable = total = 0
            adv_gold, adv_domw = [], []
            for r in brows:
                ss = r['samples'][:args.n]
                gold = r['gold']
                for _ in range(args.draws):
                    grp = rng.sample(ss, min(args.group, len(ss)))
                    ans = [s.get('ans') for s in grp]
                    rs = rfn(ans, gold)
                    total += 1
                    _, sd = stats(rs)
                    if sd > 0:
                        usable += 1
                        adv = advantages(rs)
                        cw = Counter(a for a in ans if a is not None and a != gold)
                        dom = cw.most_common(1)[0][0] if cw else None
                        for a, ad in zip(ans, adv):
                            if a == gold:
                                adv_gold.append(ad)
                            elif a == dom:
                                adv_domw.append(ad)
            ug = usable / total if total else 0
            ag = sum(adv_gold) / len(adv_gold) if adv_gold else float('nan')
            ad = sum(adv_domw) / len(adv_domw) if adv_domw else float('nan')
            summary[(bname, rname)] = (ug, ag, ad)
            print(f'{bname if rname=="exact-match" else "":16s}{rname:>10s}'
                  f'{ug:13.1%}{ag:16.3f}{ad:14.3f}')
        print()

    print('판정 기준:')
    print('  · "신호있는 그룹"이 exact-match 대비 뚜렷이 높아야 새 보상이 의미가 있다.')
    print('  · 특히 [정답없음] 구간에서 exact-match는 원리적으로 0%여야 하고,')
    print('    SC-aware가 거기서 신호를 만들어내는지가 이 설계의 핵심 주장이다.')
    print('  · [SC오답·정답있음]이 실제로 회수 가능한 구간이다 — 여기서 정답 advantage와')
    print('    지배오답 advantage가 크게 벌어져야 학습이 표차를 뒤집을 수 있다.')
    print()
    print('⚠️ 신호 존재 ≠ 학습 효과. [정답없음]에서 지배적 오답을 눌러도 올릴 정답이 없다.')


if __name__ == '__main__':
    main()
