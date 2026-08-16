"""확장 검증셋 구성 — 판정 임계를 1.49%p에서 약 0.65%p로 낮추기 위한 일회성 투자.

왜 필요한가: 우리 대응 비교(McNemar)의 판별 임계가 483문항에서 **±1.49%p**인데
성공 기준이 정확히 **+1.5%p**였다(exp65). 즉 임계선 위에 걸쳐 판정해왔고 여유가 0이었다.
부호가 일관 양수였는데 기준 미달로 기각한 것이 둘 있다 — Budget Forcing(+0.6%p, 2회 모두 양수),
검증자(+0.4~0.5%p). 개별로는 임계 아래지만 합치면 +1.1%p이고, 지금 자로는 그것도 못 잰다.

문항 수를 N배로 늘리면 임계는 %p 기준으로 1/√N배가 된다. 483 → 2,500이면 약 0.44배.

오염 검토: **최종 스택은 어댑터를 쓰지 않는다.** 베이스 모델은 우리 RFT·검증자·교사증류
데이터를 본 적이 없으므로, 그 데이터 생성에 쓴 train 문항도 베이스 스택 평가에는 오염이 아니다.
다만 **어댑터를 평가할 때는 오염이 된다.** 그래서 학습에 쓰인 id를 기록해두고,
어댑터 평가 시 제외할 수 있게 별도 컬럼으로 표시한다.

제외: ① 운영진 공개 오류 문항(627) ② 기존 검증셋 500(그대로 부분집합으로 유지하면
과거 실험과의 연속성이 깨지지 않으므로 **포함**시키고 표시만 한다)

사용: python remote/build_val_large.py --n 2500
"""
import argparse
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2500)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', default='data/val_large_ids.csv')
    args = ap.parse_args()

    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    print(f'train 전체 {len(df)}문항')

    bad_path = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    bad = set(pd.read_csv(bad_path)['id']) if os.path.exists(bad_path) else set()
    print(f'운영진 공개 오류 문항 제외: {len(bad)}')

    # 기존 검증셋(seed 123으로 뽑은 500 → 오류필터 후 483)을 그대로 부분집합으로 둔다.
    # 과거 실험 수십 건이 이 집합 위에서 판정됐으므로, 새 자에서도 같은 문항이 들어 있어야
    # "예전 결과와 새 결과가 왜 다른가"를 문항 단위로 되짚을 수 있다.
    old = df.sample(500, random_state=123).reset_index(drop=True)
    old = old[~old['id'].isin(bad)]
    old_ids = set(old['id'])
    print(f'기존 검증셋 {len(old_ids)}문항 — 부분집합으로 포함')

    pool = df[~df['id'].isin(bad | old_ids)]
    need = max(0, args.n - len(old_ids))
    extra = pool.sample(min(need, len(pool)), random_state=args.seed)
    print(f'추가 표집 {len(extra)}문항 (풀 {len(pool)}에서 seed={args.seed})')

    # 학습 데이터에 쓰인 id 표시 — 어댑터 평가 시 제외용
    used = set()
    for p, key in (('data/sft.jsonl', 'id'), ('data/sft_short.jsonl', 'id'),
                   ('data/sft_teacher_full.jsonl', 'id'), ('data/verifier.jsonl', 'id')):
        if not os.path.exists(p):
            continue
        c = 0
        for line in open(p, encoding='utf-8'):
            try:
                v = json.loads(line).get(key)
            except json.JSONDecodeError:
                continue
            if v:
                used.add(v)
                c += 1
        print(f'  학습 사용 id 수집: {p} → 누적 {len(used)}')

    sel = pd.concat([old[['id', 'answer']], extra[['id', 'answer']]])
    sel = sel.drop_duplicates('id').reset_index(drop=True)
    sel['in_old_val'] = sel['id'].isin(old_ids)
    sel['used_in_training_data'] = sel['id'].isin(used)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    sel.to_csv(args.out, index=False)

    n_used = int(sel['used_in_training_data'].sum())
    print()
    print(f'저장: {args.out} — 총 {len(sel)}문항')
    print(f'  기존 검증셋 포함     : {int(sel["in_old_val"].sum())}')
    print(f'  학습 데이터에 쓰인 것 : {n_used} ({n_used/len(sel):.1%})'
          f'  ← 베이스 평가엔 무관, 어댑터 평가 시 제외할 것')
    print()
    print(f'판정 임계 변화: 483문항 ±1.49%p → {len(sel)}문항 '
          f'±{1.49 * (483/len(sel))**0.5:.2f}%p (대응 비교 기준)')
    print('다음: dump_samples.py --val-ids ' + args.out + ' 로 베이스 표본 생성')


if __name__ == '__main__':
    main()
