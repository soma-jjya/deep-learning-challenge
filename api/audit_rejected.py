"""exp52b — 교사가 틀린 문항이 '문제가 깨진 것'인지 감사. GPU 불필요.

왜 하는가: 대상은 "베이스가 32번 시도해도 정답을 한 번도 못 만든 문제"인데, **강한 교사까지
40%를 틀린다**(exp53c: 교사 정답률 60.0%, 오답 456건). exp16이 오답 표본의 23%를 라벨 의심으로
분류한 것과 방향이 같다. 이 비율이 크면 **검증셋 자체의 신뢰도 상한**을 뜻하므로,
"로컬 85%는 도달 불가"라는 발표 논증의 직접 근거가 된다.

⚠️ 편향 주의: 판정도 같은 계열 모델이 한다. "내가 맞고 gold가 틀렸다"로 기울 수 있으므로
① 문제를 **처음부터 독립적으로 다시 풀게 하고** ② gold를 틀렸다고 하려면 근거를 명시하게 하며
③ 애매하면 TEACHER_WRONG으로 기울도록 지시한다. 그래도 남는 편향은 결과 해석 시 함께 말할 것.

사용: python api/audit_rejected.py --n 30
"""
import argparse
import json
import os
import random
import re
import subprocess

NL = chr(10)

SYSTEM = (
    '당신은 수학 문제의 품질을 감사하는 심사자입니다. 어떤 강한 모델이 낸 답이 정답 라벨(gold)과 '
    '달랐던 사례를 봅니다. 판정할 것은 "누가 더 똑똑한가"가 아니라 **문제와 라벨이 온전한가**입니다.' + NL * 2 +
    '절차:' + NL +
    '1. 제시된 풀이를 읽기 전에, 문제를 **처음부터 스스로** 푸세요.' + NL +
    '2. 그 다음 gold와 비교하세요.' + NL +
    '3. 아래 한 가지로 분류하세요.' + NL * 2 +
    '- GOLD_WRONG : 문제는 명확한데 gold 값이 틀렸다. **당신 자신의 독립 풀이가 gold와 다르고 '
    '그 근거를 댈 수 있을 때만** 사용하세요.' + NL +
    '- AMBIGUOUS : 문제 문장이 모호하거나 정보가 빠져 답이 하나로 정해지지 않는다 '
    '(해석에 따라 답이 갈리는 경우 포함).' + NL +
    '- TEACHER_WRONG : 문제와 gold는 온전하고 제시된 풀이가 틀렸다.' + NL * 2 +
    '**확신이 서지 않으면 TEACHER_WRONG을 고르세요.** 라벨을 의심하는 쪽이 편한 판정이므로 '
    '기준을 높게 잡습니다.' + NL * 2 +
    '출력은 정확히 이 두 줄만:' + NL +
    'VERDICT: <GOLD_WRONG|AMBIGUOUS|TEACHER_WRONG>' + NL +
    'REASON: <한 문장>' + NL
)

VERDICT_PAT = re.compile(r'VERDICT:\s*(GOLD_WRONG|AMBIGUOUS|TEACHER_WRONG)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rejected', default='data/teacher_cc_rejected.jsonl')
    ap.add_argument('--src', default='data/hard_problems_top.csv')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='results/audit_rejected.jsonl')
    ap.add_argument('--timeout', type=int, default=600)
    args = ap.parse_args()

    import csv
    csv.field_size_limit(10 ** 7)
    q = {r['id']: r['question'] for r in csv.DictReader(open(args.src, encoding='utf-8'))}

    rows = [json.loads(l) for l in open(args.rejected, encoding='utf-8')]
    rows = [r for r in rows if r['id'] in q]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]
    print(f'감사 대상 {len(rows)}건 (전체 오답 중 무작위 표본, seed={args.seed})')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    done = {}
    if os.path.exists(args.out):
        for l in open(args.out, encoding='utf-8'):
            d = json.loads(l)
            done[d['id']] = d['verdict']
        print(f'  이미 판정됨 {len(done)}건 — 건너뜀')

    counts = {}
    for v in done.values():
        counts[v] = counts.get(v, 0) + 1

    with open(args.out, 'a', encoding='utf-8') as w:
        for i, r in enumerate(rows):
            if r['id'] in done:
                continue
            prompt = (SYSTEM + NL + '---' + NL * 2 +
                      '문제:' + NL + q[r['id']].strip() + NL * 2 +
                      f'정답 라벨(gold): {r["gold"]}' + NL * 2 +
                      f'제시된 풀이가 낸 답: {r["teacher_answer"]}' + NL * 2 +
                      '제시된 풀이:' + NL + str(r['solution'])[:6000] + NL)
            try:
                p = subprocess.run(
                    ['claude', '-p', prompt, '--max-turns', '1',
                     '--dangerously-skip-permissions'],
                    capture_output=True, text=True, timeout=args.timeout)
                m = VERDICT_PAT.search(p.stdout or '')
            except subprocess.TimeoutExpired:
                m = None
            if not m:
                print(f'  [{r["id"]}] 판정 실패 — 건너뜀', flush=True)
                continue
            v = m.group(1)
            reason = ''
            rm = re.search(r'REASON:\s*(.+)', p.stdout or '')
            if rm:
                reason = rm.group(1).strip()[:300]
            w.write(json.dumps({'id': r['id'], 'verdict': v, 'reason': reason,
                                'gold': r['gold'], 'teacher_answer': r['teacher_answer']},
                               ensure_ascii=False) + NL)
            w.flush()
            counts[v] = counts.get(v, 0) + 1
            print(f'  [{i+1}/{len(rows)}] {r["id"]} → {v}', flush=True)

    total = sum(counts.values())
    print()
    print(f'판정 완료 {total}건')
    for k in ('GOLD_WRONG', 'AMBIGUOUS', 'TEACHER_WRONG'):
        c = counts.get(k, 0)
        print(f'  {k:15s} {c:3d}  ({c/max(1,total):.1%})')
    bad = counts.get('GOLD_WRONG', 0) + counts.get('AMBIGUOUS', 0)
    print()
    print(f'문제·라벨 결함 의심: {bad}/{total} = {bad/max(1,total):.1%}')
    print('⚠️ 판정도 같은 계열 모델이 하므로 라벨을 의심하는 쪽으로 기울 수 있다 — 상한으로 읽을 것.')


if __name__ == '__main__':
    main()
