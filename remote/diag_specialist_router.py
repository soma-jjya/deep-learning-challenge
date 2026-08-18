"""유형 기반 specialist router의 **배포 가능 이득**을 시드 홀드아웃으로 잰다.

diag_specialist_localization.py에서 유형x계열 표를 얻었지만, 그 표를 눈으로 보고
"여기가 좋아 보인다"를 고르면 그건 선택 후 평가다 — exp55c에서 nested CV를 붙이자마자
+4가 -1.4로 뒤집힌 것과 정확히 같은 함정이다.

그래서 여기서는 **규칙 선택과 평가에 서로 다른 시드를 쓴다**:
  · 선택 시드 k 하나에서 (유형, 계열) 칸의 순효용을 재고, 양수인 칸만 override 규칙으로 채택
  · 나머지 두 시드에서 그 규칙을 그대로 적용해 베이스 대비 순변화를 측정
  · 세 가지 선택 시드에 대해 반복해 평균낸다 (leave-one-seed-out)

기본 정책은 보수적이다: default = base, 채택된 칸에 걸린 문항만 specialist로 override.
칸이 여러 개 걸리면 선택 시드에서 순효용이 가장 컸던 칸을 쓴다.

민감도: 채택 임계(칸의 순효용 최소값)를 0,1,2,3문항으로 훑는다.

사용: PYTHONIOENCODING=utf-8 python remote/diag_specialist_router.py
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_specialist_localization import BASE, FAMILIES, TRAIN, feats, load  # noqa: E402

csv.field_size_limit(10 ** 7)


def main():
    with open(TRAIN, encoding='utf-8') as f:
        qmap = {r['id']: r['question'] for r in csv.DictReader(f)}

    base = [load(p) for p in BASE]
    ids = sorted(set(base[0]) & set(base[1]) & set(base[2]))
    fams = {}
    for fam, paths in FAMILIES.items():
        s = [load(p) for p in paths]
        if all(x is not None for x in s):
            fams[fam] = s
            ids = [i for i in ids if all(i in x for x in s)]

    F = {i: feats(qmap.get(i, '')) for i in ids}
    names = list(next(iter(F.values())).keys())
    cells = [(fam, nm) for fam in fams for nm in names
             if sum(1 for i in ids if F[i][nm]) >= 15]

    print('검증 %d문항 · 계열 %d개 · 유형 %d개 · 후보 칸 %d개'
          % (len(ids), len(fams), len(names), len(cells)))
    print('베이스 정확도(시드42/43/44): %s' % [sum(b[i] for i in ids) for b in base])
    print()

    def net(fam, nm, k):
        """선택 시드 k에서 (계열, 유형) 칸의 순효용 = 구제 - 훼손"""
        return sum(fams[fam][k][i] - base[k][i] for i in ids if F[i][nm])

    for thr in (0, 1, 2, 3):
        print('=' * 74)
        print('채택 임계: 선택 시드에서 칸 순효용 >= +%d 문항' % max(thr, 1))
        outs = []
        for ksel in range(3):
            chosen = [(fam, nm, net(fam, nm, ksel)) for fam, nm in cells
                      if net(fam, nm, ksel) >= max(thr, 1)]
            chosen.sort(key=lambda x: -x[2])
            # 문항별 override 대상 계열 결정 (순효용 큰 칸 우선)
            route = {}
            for fam, nm, v in chosen:
                for i in ids:
                    if F[i][nm] and i not in route:
                        route[i] = fam
            deltas = []
            for kev in range(3):
                if kev == ksel:
                    continue
                d = sum(fams[route[i]][kev][i] - base[kev][i] for i in route)
                deltas.append(d)
            outs.extend(deltas)
            print('  선택=시드%d: 채택 칸 %2d개, override 문항 %3d개 → 홀드아웃 순변화 %s'
                  % (42 + ksel, len(chosen), len(route), deltas))
            if chosen:
                print('     채택: %s' % ', '.join('%s/%s(+%d)' % (f, n, v)
                                                 for f, n, v in chosen[:6]))
        if outs:
            print('  ▶ 홀드아웃 6회 평균 순변화: %+.2f 문항  (양수 %d/6)'
                  % (sum(outs) / len(outs), sum(1 for x in outs if x > 0)))
        print()

    # 상한 참고: 만약 칸 선택이 완벽하다면(=평가 시드에서 직접 최적 칸을 고른다면)
    print('=' * 74)
    best = 0
    for kev in range(3):
        g = 0
        for fam, nm in cells:
            v = sum(fams[fam][kev][i] - base[kev][i] for i in ids if F[i][nm])
            g = max(g, v)
        best += g
    print('참고 상한: 평가 시드에서 직접 최고 칸 하나를 고를 때 %+.1f 문항/시드'
          % (best / 3))
    print('  (이 값은 선택 후 평가라 배포 불가능한 낙관치다 — 위 홀드아웃과의 격차가 곧 과적합량)')


if __name__ == '__main__':
    main()
