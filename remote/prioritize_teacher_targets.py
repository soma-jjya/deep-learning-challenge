"""교사 증류 대상 우선순위 재선정 (H23 개정) — 분야별 진단 결과 반영. GPU 불필요.

당초 대상은 "베이스가 RFT에서 못 푼 문제" 2,967개 전부였다. 그런데 분야별로 재보니
성격이 완전히 다르다(검증 483문항 기준):

  분야            정확도   정답부재율   ← 32번 풀어도 정답이 한 번도 안 나오는 비율
  기타/문장제      85.5%     6.0%
  기하            56.3%    24.4%
  정수론          54.3%    25.9%
  조합론          30.0%    40.0%

문장제는 틀려도 **정답 후보는 만들어낸다**(완전 실패 6%). 반면 경시형은 20~40%가
아예 정답을 생성하지 못한다 — 즉 정밀도 문제가 아니라 **능력 부재**다.
교사 풀이가 메울 수 있는 것은 후자이므로, 같은 예산이면 경시형에 쓰는 편이 밀도가 높다.

또 하나: 어려운 문제 집합은 자동생성 역문제("If we know the answer is X...")가
전체 대비 4.6배 농축돼 있다. 이런 문제는 라벨이 부정확하기 쉬워 학습 데이터로 부적합하므로
우선순위를 낮춘다(완전 제외는 하지 않는다 — gold 대조 필터가 어차피 걸러낸다).

사용: python remote/prioritize_teacher_targets.py --top 1200
출력: data/hard_problems_prioritized.csv (category, priority 컬럼 추가)
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from analyze_by_category import classify

csv.field_size_limit(10 ** 7)

# 검증셋에서 측정한 분야별 '정답부재율' — 교사 풀이의 한계 가치가 높은 순
NO_CORRECT_RATE = {
    '조합론': 0.400, '수열/급수': 0.294, '정수론': 0.259, '기하': 0.244,
    '대수/방정식': 0.207, '확률/통계': 0.200, '함수': 0.286, '미적분': 0.667,
    '기타/문장제': 0.060,
}
# 표본이 작은 분야는 추정이 불안정하므로 상한을 씌워 과도한 가중을 막는다
CAP = 0.30

DEGENERATE = re.compile(r'if we know the answer|unknown variable\s+[A-Za-z]', re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/hard_problems.csv')
    ap.add_argument('--out', default='data/hard_problems_prioritized.csv')
    ap.add_argument('--top', type=int, default=None,
                    help='상위 N개만 별도 저장 (예산이 제한될 때)')
    ap.add_argument('--top-out', default='data/hard_problems_top.csv')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.src, encoding='utf-8')))
    for r in rows:
        cat = classify(r['question'])
        r['category'] = cat
        score = min(NO_CORRECT_RATE.get(cat, 0.2), CAP)
        if DEGENERATE.search(r['question'] or ''):
            score -= 0.5          # 자동생성 역문제는 뒤로
        r['priority'] = f'{score:.4f}'

    rows.sort(key=lambda r: -float(r['priority']))
    cols = ['id', 'question', 'answer', 'category', 'priority']
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: r[k] for k in cols} for r in rows)
    print(f'전체 {len(rows)}문제 우선순위 부여 → {args.out}')

    if args.top:
        top = rows[:args.top]
        with open(args.top_out, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows({k: r[k] for k in cols} for r in top)
        c = Counter(r['category'] for r in top)
        deg = sum(1 for r in top if DEGENERATE.search(r['question'] or ''))
        print(f'\n상위 {len(top)}문제 → {args.top_out}')
        print(f'{"분야":14s}{"문항":>7s}{"비중":>8s}')
        print('-' * 32)
        for k, v in c.most_common():
            print(f'{k:14s}{v:>7d}{v/len(top):>8.1%}')
        print(f'\n자동생성 역문제 포함: {deg}개 ({deg/len(top):.1%}) — 전체 집합의 농도보다 낮아야 정상')
    print()
    print('다음: api/make_desktop_chunks.py --src <위 파일> 또는')
    print('      api/gen_teacher_solutions.py --src <위 파일> (API 키 필요)')


if __name__ == '__main__':
    main()
