"""교사 풀이 생성을 Claude Desktop에서 수행하기 위한 문제 묶음 파일 생성. GPU·API 불필요.

배치 API 대신 Desktop에서 만들 때의 제약:
  - 한 번에 넣을 수 있는 문제 수가 제한적 → 묶음으로 쪼갠다
  - 응답을 다시 파일로 되돌려야 함 → 파싱 가능한 고정 형식을 강제한다

출력:
  data/desktop/problems_001.md ...   : Desktop에 붙여넣을 문제 묶음
  data/desktop/INDEX.md              : 진행 체크리스트 (어디까지 했는지 표시용)

사용: python api/make_desktop_chunks.py --chunk 20
"""
import argparse
import os

import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/hard_problems.csv')
    ap.add_argument('--out-dir', default='data/desktop')
    ap.add_argument('--chunk', type=int, default=20, help='묶음당 문제 수')
    ap.add_argument('--limit', type=int, default=None, help='앞에서 N개만 (파일럿용)')
    args = ap.parse_args()

    with open(args.src, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    nl = chr(10)
    n_chunks = (len(rows) + args.chunk - 1) // args.chunk

    for i in range(n_chunks):
        part = rows[i * args.chunk:(i + 1) * args.chunk]
        path = os.path.join(args.out_dir, f'problems_{i+1:03d}.md')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'# 묶음 {i+1}/{n_chunks} — {len(part)}문제' + nl * 2)
            f.write('아래 각 문제를 풀고, 지정된 형식으로만 출력하세요.' + nl * 2)
            for r in part:
                f.write('---' + nl * 2)
                f.write(f'## ID: {r["id"]}' + nl * 2)
                f.write(r['question'].strip() + nl * 2)

    # 정답은 별도 파일로 분리 — 교사에게 답을 알려주면 풀이가 아니라 역산을 하게 된다
    gold_path = os.path.join(args.out_dir, 'gold.csv')
    with open(gold_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'answer'])
        for r in rows:
            w.writerow([r['id'], r['answer']])

    with open(os.path.join(args.out_dir, 'INDEX.md'), 'w', encoding='utf-8') as f:
        f.write(f'# 진행 체크리스트 — 총 {n_chunks}묶음 / {len(rows)}문제' + nl * 2)
        f.write('한 묶음 끝낼 때마다 응답 전체를 `data/desktop/out_NNN.txt`로 저장하고 아래에 체크.' + nl * 2)
        for i in range(n_chunks):
            f.write(f'- [ ] problems_{i+1:03d}.md → out_{i+1:03d}.txt' + nl)

    print(f'문제 {len(rows)}개 → {n_chunks}묶음 (묶음당 {args.chunk}개)')
    print(f'저장: {args.out_dir}/problems_*.md')
    print(f'정답: {gold_path}  ← Desktop에 넣지 말 것 (채점 전용)')
    print(f'체크리스트: {args.out_dir}/INDEX.md')


if __name__ == '__main__':
    main()
