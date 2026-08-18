"""단일 칸 집중 검정 — "방정식/다항식 문항은 마스킹 재학습 계열에 맡긴다"

diag_specialist_router.py의 홀드아웃은 전체적으로 음수(-0.33 ~ -0.83문항)였지만,
**한 칸만 세 선택 시드 전부에서, 그리고 서로 다른 두 계열에서 반복**해 나왔다:

    유형 '방정식/다항식' (26문항)
      rftmask (RFT 데이터 마스킹 재학습) : 구제 [2,1,1] / 훼손 [0,0,0]
      masked  (교사 데이터 마스킹 재학습) : 구제 [2,2,1] / 훼손 [0,0,0]

6번의 독립 시드 실행에서 **훼손이 한 번도 없었다.** 두 계열은 학습 데이터가 서로 다르다
(exp59는 교사 풀이, exp60은 RFT 풀이) — 레시피만 공유한다.

다만 이 칸은 **65개 칸을 눈으로 보고 고른 것**이다. 그래서 여기서 하는 일은 두 가지다.
  (1) 다중비교 보정된 순열검정 — 65칸 전체에서 이만한 칸이 우연히 나올 확률
  (2) 이 칸만 쓰는 라우터의 leave-one-seed-out 배포 순변화

사용: PYTHONIOENCODING=utf-8 python remote/diag_equation_cell.py
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_specialist_localization import BASE, FAMILIES, TRAIN, feats, load  # noqa: E402

csv.field_size_limit(10 ** 7)
random.seed(1)

CELL = '방정식/다항식'
TARGETS = ['rftmask', 'masked']


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
    feat_ids = {nm: [i for i in ids if F[i][nm]] for nm in names}
    cells = [(fam, nm) for fam in fams for nm in names if len(feat_ids[nm]) >= 15]

    sub = feat_ids[CELL]
    print('유형 %s: %d문항 / 전체 %d' % (CELL, len(sub), len(ids)))
    print('베이스 이 구간 정확도(시드42/43/44): %s'
          % [sum(base[k][i] for i in sub) for k in range(3)])
    print()

    def cellnet(fam, nm, k):
        return sum(fams[fam][k][i] - base[k][i] for i in feat_ids[nm])

    for fam in TARGETS:
        n = [cellnet(fam, CELL, k) for k in range(3)]
        print('  %-8s 순효용 %s  (합 %+d)' % (fam, n, sum(n)))
    combined = sum(cellnet(f, CELL, k) for f in TARGETS for k in range(3))
    print('  두 계열 6회 합계 %+d문항 (시드당 평균 %+.2f)' % (combined, combined / 6))
    print()

    # (1) 다중비교 보정 순열검정 —— 65칸 중 최고가 이만큼 나올 확률
    print('=' * 70)
    obs_single = max(sum(cellnet(f, CELL, k) for k in range(3)) for f in TARGETS)
    TRIALS = 3000
    hit_single = hit_allpos = 0
    pool = list(ids)
    for _ in range(TRIALS):
        perm = pool[:]
        random.shuffle(perm)
        m = dict(zip(pool, perm))
        best = -99
        best_allpos = False
        for fam, nm in cells:
            ss = [m[i] for i in feat_ids[nm]]
            per_seed = [sum(fams[fam][k][i] - base[k][i] for i in ss) for k in range(3)]
            v = sum(per_seed)
            if v > best:
                best = v
                best_allpos = all(x > 0 for x in per_seed)
        if best >= obs_single:
            hit_single += 1
        if best >= obs_single and best_allpos:
            hit_allpos += 1
    print('순열검정(65칸 최댓값 기준, %d회):' % TRIALS)
    print('  관측 최고 단일칸 3시드 합 = %+d' % obs_single)
    print('  우연히 이 이상 = %.3f   그중 3시드 모두 양수까지 = %.3f'
          % (hit_single / TRIALS, hit_allpos / TRIALS))
    print('  → 65칸을 훑으면 이 정도 칸은 우연으로도 흔하게 나온다는 뜻이면 채택 불가')
    print()

    # (2) 이 칸만 쓰는 라우터의 leave-one-seed-out 배포 순변화
    print('=' * 70)
    print('이 칸만 쓰는 라우터 (default=base, 방정식/다항식 문항만 override):')
    for fam in TARGETS:
        outs = []
        for ksel in range(3):
            if cellnet(fam, CELL, ksel) <= 0:
                sel = False
            else:
                sel = True
            for kev in range(3):
                if kev == ksel:
                    continue
                outs.append(cellnet(fam, CELL, kev) if sel else 0)
        print('  %-8s 홀드아웃 6회 %s → 평균 %+.2f문항 (양수 %d/6)'
              % (fam, outs, sum(outs) / 6, sum(1 for x in outs if x > 0)))
    print()
    print('=' * 70)
    print('배포 관점 판정 기준:')
    print('  · 이 칸이 진짜여도 시드당 기대 이득은 1~2문항 = 0.2~0.4%p')
    print('  · 리더보드 재실행 변동은 실측 0.72%%p (exp71) — 이득이 노이즈의 절반 이하')
    print('  → 통계적으로 살아남더라도 8/31 최종 스택 변경 근거로는 부족하다')


if __name__ == '__main__':
    main()
