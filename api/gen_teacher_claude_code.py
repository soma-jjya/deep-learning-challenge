"""교사 풀이 생성 — 서버에 이미 있는 Claude Code를 헤드리스로 호출. 추가 비용 0, GPU 불필요.

배치 API(별도 키 필요) 대신 `claude -p`를 쓴다. 러너가 이미 같은 방식으로 돌고 있고
토큰도 `~/.ajudl_env`에 있으므로 새로 준비할 것이 없다.

규칙 근거: prd.md:30 — 학습 데이터 구축 목적의 상용 모델 사용은 허용.
train 문제만 대상이며(prd.md:31 준수), 추론 단계에서는 어떤 외부 모델도 쓰지 않는다.

출력은 `api/parse_desktop_output.py`가 그대로 읽을 수 있는 형식(`## ID: xxx` 블록)으로
저장한다 — 이미 왕복 테스트를 통과한 파서를 재사용하기 위해서다.

사용:
    python api/gen_teacher_claude_code.py --src data/hard_problems_top.csv --limit 40   # 파일럿
    python api/gen_teacher_claude_code.py --src data/hard_problems_top.csv              # 전량
    python api/parse_desktop_output.py --in-glob 'data/teacher_out/out_*.txt' --gold data/desktop_top/gold.csv
"""
import argparse
import csv
import os
import subprocess
import sys
import time

csv.field_size_limit(10 ** 7)
BS = chr(92)
NL = chr(10)

SYSTEM = (
    '당신은 올림피아드 수준의 수학자입니다. 당신이 쓰는 풀이는 사람이 읽으려는 것이 아니라, '
    '작은 언어 모델(3B)이 학습해서 따라 하기 위한 교재입니다.' + NL * 2 +
    '규칙:' + NL +
    '- 단계를 건너뛰지 마세요. 산술 한 줄도 생략하지 않습니다.' + NL +
    '- 코드 실행·도구·파일 읽기·웹 검색을 절대 쓰지 마세요. 순수한 수학적 추론만 사용합니다. '
    '학생 모델은 오프라인에서 추론만으로 풀어야 하므로 도구에 의존한 풀이는 학습 가치가 없습니다.' + NL +
    '- 서론·자기 언급·총평 없이 바로 풀이로 들어갑니다.' + NL +
    '- 최종 답은 항상 정수입니다.' + NL +
    '- 못 풀겠어도 건너뛰지 말고 최선의 풀이와 답을 내세요. 틀린 답은 자동으로 걸러집니다.' + NL * 2 +
    '출력 형식(이 형식만 지킬 것 — 자동 파싱됩니다). 문제마다 이 블록을 반복하세요:' + NL * 2 +
    '## ID: <문제 ID 그대로>' + NL * 2 +
    '<풀이 본문>' + NL * 2 +
    BS + 'boxed{정수}' + NL
)


def build_prompt(batch):
    parts = [SYSTEM, NL, f'아래 {len(batch)}문제를 위 형식대로 푸세요.', NL]
    for r in batch:
        parts.append(NL + '---' + NL * 2 + f'## ID: {r["id"]}' + NL * 2 + r['question'].strip() + NL)
    return ''.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/hard_problems_top.csv')
    ap.add_argument('--out-dir', default='data/teacher_out')
    ap.add_argument('--batch', type=int, default=5, help='한 번의 호출에 넣을 문제 수')
    ap.add_argument('--limit', type=int, default=None, help='앞에서 N문제만 (파일럿)')
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--timeout', type=int, default=900, help='호출당 최대 대기(초)')
    ap.add_argument('--retries', type=int, default=1)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.src, encoding='utf-8')))[args.offset:]
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    n_batches = (len(rows) + args.batch - 1) // args.batch
    print(f'대상 {len(rows)}문제 → {n_batches}회 호출 (배치 {args.batch})')
    done = ok = 0
    t0 = time.time()

    for b in range(n_batches):
        idx = args.offset // args.batch + b
        out_path = os.path.join(args.out_dir, f'out_{idx:04d}.txt')
        if os.path.exists(out_path) and os.path.getsize(out_path) > 200:
            done += 1
            continue                     # 이미 만든 것은 건너뛴다(재개 가능)

        batch = rows[b * args.batch:(b + 1) * args.batch]
        prompt = build_prompt(batch)
        for attempt in range(args.retries + 1):
            try:
                # --max-turns 1: 도구를 쓰지 말고 한 번에 답만 내게 한다
                p = subprocess.run(
                    ['claude', '-p', prompt, '--max-turns', '1',
                     '--dangerously-skip-permissions'],
                    capture_output=True, text=True, timeout=args.timeout)
                text = p.stdout or ''
                if '## ID:' in text:
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    ok += 1
                    break
                print(f'  [{idx}] ID 헤더 없음 (시도 {attempt+1}) — stderr: {(p.stderr or "")[:120]}')
            except subprocess.TimeoutExpired:
                print(f'  [{idx}] 시간 초과 (시도 {attempt+1})')
        done += 1
        if done % 5 == 0 or done == n_batches:
            el = time.time() - t0
            rate = el / max(1, done)
            print(f'진행 {done}/{n_batches}배치, 성공 {ok} — '
                  f'경과 {el/60:.1f}분, 예상 잔여 {(n_batches-done)*rate/60:.1f}분', flush=True)

    print()
    print(f'완료: {ok}/{n_batches}배치 성공 → {args.out_dir}/out_*.txt')
    print('다음: python api/parse_desktop_output.py '
          f"--in-glob '{args.out_dir}/out_*.txt' --gold data/desktop_top/gold.csv --tag cc")


if __name__ == '__main__':
    main()
