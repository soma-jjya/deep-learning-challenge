"""[보조 경로] 배치 API로 교사 풀이 생성 — 별도 API 키가 있을 때만 쓴다.

기본 경로는 `api/gen_teacher_claude_code.py`다. 서버에 이미 있는 Claude Code를
헤드리스(`claude -p`)로 호출하므로 별도 키가 필요 없다. 이 파일은 대량 병렬 처리량이
필요해질 때를 위한 대안으로만 남겨둔다.

(H23) 학습 데이터 구축. GPU 불필요.

배경: 학습 6전(SFT×3, GRPO×2, DPO×1)이 전부 실패했는데, **6전 모두 학습 데이터가
자기 자신이 만든 풀이(RFT) 아니면 외부 공개 데이터(NuminaMath)였다.** 즉 "모델이 이미
아는 것"을 다시 먹였거나, 우리 문제 분포와 무관한 데이터를 섞었다. 강한 교사가
**우리 문제에 대해** 만든 풀이는 한 번도 써본 적이 없다 — 이것이 학습 축에서 유일하게
남은 미탐색 변형이다.

대상: `data/hard_problems.csv` (베이스가 RFT에서 정답에 도달하지 못한 문제).
RFT가 구조적으로 만들 수 없는 데이터라 자기증류의 한계를 정면으로 벗어난다.

품질 보증: 교사 풀이 중 **최종 답이 gold와 일치하는 것만** 채택한다 → 라벨 노이즈 0.

규칙 근거: prd.md:30 학습 데이터 구축 목적의 상용 API 사용 허용.
  - train 문제만 사용 (prd.md:31 테스트 문제 답 생성 금지 준수)
  - 추론 시에는 API를 쓰지 않는다 (prd.md:35 오프라인 추론 원칙) — 이 스크립트의 산출물은
    학습 데이터일 뿐이며, 최종 제출 추론은 로컬 vLLM 단독으로 수행
  - 최종 제출 시 '사용 데이터 목록'에 API 사용 사실을 명시할 것 (prd.md:94)

사용:
    export ANTHROPIC_API_KEY=...
    python api/gen_teacher_solutions.py --limit 300 --tag pilot     # 파일럿 (비용·수율 측정)
    python api/gen_teacher_solutions.py --tag full                  # 전량
"""
import argparse
import json
import os
import re
import sys
import time

BS = chr(92)
NUM_PAT = re.compile('-?[0-9][0-9,]*(?:[.][0-9]+)?')
BOXED_PAT = re.compile('boxed *{((?:[^{}]|{[^{}]*})*)}')

SYSTEM = (
    'You are an expert olympiad mathematician writing a reference solution that a small '
    'language model will learn from. Solve the problem with clear, complete, step-by-step '
    'reasoning that can be followed without leaps. Keep every arithmetic step explicit. '
    'Do not use code, tools, or external references — plain mathematical reasoning only, '
    'because the student model must be able to reproduce this offline. '
    'The final answer is always an integer. '
    'End with the final integer answer inside ' + BS + 'boxed{}.')


def extract_answer(text):
    """교사 풀이에서 최종 정수 답 추출 — 채택 판정용(gold 대조)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/hard_problems.csv')
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--tag', default='pilot')
    ap.add_argument('--model', default='claude-opus-5')
    ap.add_argument('--effort', default='high', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    ap.add_argument('--max-tokens', type=int, default=8000)
    ap.add_argument('--limit', type=int, default=None, help='앞에서 N개만 (파일럿용)')
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--poll', type=int, default=60, help='배치 상태 확인 주기(초)')
    args = ap.parse_args()

    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    import pandas as pd

    df = pd.read_csv(args.src)
    df = df.iloc[args.offset:]
    if args.limit:
        df = df.iloc[:args.limit]
    print(f'대상 {len(df)}문제 ({args.src}, offset={args.offset}), 교사={args.model}, effort={args.effort}')

    client = anthropic.Anthropic()

    requests = [
        Request(
            custom_id=str(row.id),
            params=MessageCreateParamsNonStreaming(
                model=args.model,
                max_tokens=args.max_tokens,
                system=SYSTEM,
                thinking={'type': 'adaptive'},
                output_config={'effort': args.effort},
                messages=[{'role': 'user', 'content': str(row.question)}],
            ),
        )
        for row in df.itertuples()
    ]

    batch = client.messages.batches.create(requests=requests)
    print(f'배치 생성: {batch.id} — 대개 1시간 내, 최대 24시간')

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, f'teacher_batch_{args.tag}.json')
    json.dump({'batch_id': batch.id, 'model': args.model, 'effort': args.effort,
               'n_requests': len(requests), 'src': args.src,
               'offset': args.offset, 'limit': args.limit},
              open(meta_path, 'w'), indent=2)
    print(f'배치 정보 저장: {meta_path} (중단 시 --resume-batch로 이어받기 가능)')

    while True:
        b = client.messages.batches.retrieve(batch.id)
        c = b.request_counts
        print(f'  [{time.strftime("%H:%M:%S")}] {b.processing_status} — '
              f'처리중 {c.processing} / 성공 {c.succeeded} / 실패 {c.errored}')
        if b.processing_status == 'ended':
            break
        time.sleep(args.poll)

    gold = {str(row.id): int(row.answer) for row in df.itertuples()}
    out_path = os.path.join(args.out_dir, f'teacher_{args.tag}.jsonl')
    kept = dropped_wrong = dropped_noans = errored = 0
    in_tok = out_tok = 0

    with open(out_path, 'w', encoding='utf-8') as f:
        for res in client.messages.batches.results(batch.id):
            if res.result.type != 'succeeded':
                errored += 1
                continue
            msg = res.result.message
            in_tok += msg.usage.input_tokens
            out_tok += msg.usage.output_tokens
            text = ''.join(b.text for b in msg.content if b.type == 'text')
            ans = extract_answer(text)
            if ans is None:
                dropped_noans += 1
                continue
            if ans != gold.get(res.custom_id):
                dropped_wrong += 1      # 교사도 틀린 문제 — 라벨 노이즈가 되므로 버린다
                continue
            f.write(json.dumps({'id': res.custom_id, 'solution': text.strip(),
                                'answer': ans}, ensure_ascii=False) + chr(10))
            kept += 1

    total = kept + dropped_wrong + dropped_noans
    print()
    print(f'채택(정답 일치)   : {kept}')
    print(f'교사도 오답        : {dropped_wrong}')
    print(f'답 추출 실패       : {dropped_noans}')
    print(f'API 오류           : {errored}')
    if total:
        print(f'교사 정답률        : {kept/total:.1%}  ← 이 값이 낮으면 이 트랙의 상한도 낮다')
    print(f'토큰: 입력 {in_tok:,} / 출력 {out_tok:,}')
    print(f'저장: {out_path}')
    print()
    print('다음: python api/build_sft_from_teacher.py 로 학습 형식(messages)으로 변환')


if __name__ == '__main__':
    main()
