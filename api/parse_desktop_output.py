"""Claude Desktop 응답 → 교사 풀이 JSONL 변환 + gold 대조 채점. GPU·API 불필요.

`data/desktop/out_*.txt`에 저장한 Desktop 응답들을 읽어, 배치 API 경로와 **완전히 동일한
출력 형식**(data/teacher_*.jsonl)으로 만든다. 이후 단계(build_sft_from_teacher.py)는
어느 경로로 만들었든 그대로 쓸 수 있다.

채점 원칙: 최종 답이 gold와 일치하는 풀이만 채택 → 라벨 노이즈 0.
(교사도 틀릴 수 있고, 틀린 풀이를 학습시키면 자기증류보다 나쁘다)

사용: python api/parse_desktop_output.py --tag desktop
"""
import argparse
import glob
import json
import os
import re

BS = chr(92)
NUM_PAT = re.compile('-?[0-9][0-9,]*(?:[.][0-9]+)?')
BOXED_PAT = re.compile('boxed *{((?:[^{}]|{[^{}]*})*)}')
# 응답 구분자: "### ID: train-000123" 또는 "## ID: train-000123"
ID_PAT = re.compile(r'^#{2,4}\s*ID\s*[:：]\s*([A-Za-z0-9_-]+)\s*$', re.MULTILINE)


def extract_answer(text):
    m = BOXED_PAT.findall(text.replace(BS, ''))
    cand = m[-1] if m else None
    if cand is None:
        nums = NUM_PAT.findall(text)
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    n = NUM_PAT.findall(cand)
    if not n:
        return None
    try:
        return int(round(float(n[-1].replace(',', ''))))
    except ValueError:
        return None


def split_by_id(text):
    """응답 전체를 ID 헤더 기준으로 잘라 (id, 풀이본문) 목록으로."""
    hits = list(ID_PAT.finditer(text))
    out = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        body = text[m.end():end].strip()
        if body:
            out.append((m.group(1), body))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-glob', default='data/desktop/out_*.txt')
    ap.add_argument('--gold', default='data/desktop/gold.csv')
    ap.add_argument('--out', default=None, help='기본 data/teacher_<tag>.jsonl')
    ap.add_argument('--tag', default='desktop')
    args = ap.parse_args()

    import csv
    gold = {}
    with open(args.gold, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            gold[r['id']] = int(r['answer'])

    files = sorted(glob.glob(args.in_glob))
    if not files:
        raise SystemExit(f'응답 파일이 없습니다: {args.in_glob}')

    out_path = args.out or f'data/teacher_{args.tag}.jsonl'
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    seen = set()
    kept = wrong = noans = unknown = dup = 0
    with open(out_path, 'w', encoding='utf-8') as w:
        for path in files:
            text = open(path, encoding='utf-8').read()
            pairs = split_by_id(text)
            if not pairs:
                print(f'  ⚠️ {os.path.basename(path)}: ID 헤더를 찾지 못함 — 형식 확인 필요')
            for pid, body in pairs:
                if pid not in gold:
                    unknown += 1
                    continue
                if pid in seen:
                    dup += 1
                    continue
                seen.add(pid)
                ans = extract_answer(body)
                if ans is None:
                    noans += 1
                    continue
                if ans != gold[pid]:
                    wrong += 1
                    continue
                w.write(json.dumps({'id': pid, 'solution': body, 'answer': ans},
                                   ensure_ascii=False) + chr(10))
                kept += 1

    total = kept + wrong + noans
    print(f'응답 파일 {len(files)}개 처리')
    print(f'  채택(정답 일치) : {kept}')
    print(f'  교사도 오답      : {wrong}')
    print(f'  답 추출 실패     : {noans}')
    print(f'  gold에 없는 ID   : {unknown}')
    print(f'  중복 ID(무시)    : {dup}')
    if total:
        print(f'  교사 정답률      : {kept/total:.1%}  ← 이 값이 이 트랙의 상한을 결정')
    print(f'저장: {out_path}')
    print()
    print('다음: python api/build_sft_from_teacher.py --teacher ' + out_path)


if __name__ == '__main__':
    main()
