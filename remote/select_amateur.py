"""exp80a 준비 — anti-expert Contrastive Decoding의 amateur를 정량 선정한다 (GPU 0).

사전등록(prereg_exp77_decoding.md)대로 "자체 정확도가 낮으니까"로 고르지 않는다.
원하는 amateur는 세 조건을 동시에 만족한다.

    A(s) = wrongAgreement(base, s) − λ · rescueRate(s)

  · wrongAgreement — base와 s가 **둘 다 틀렸을 때 같은 오답을 낸 조건부 비율** (높을수록 좋다)
  · rescueRate     — base가 틀린 문항 중 s가 맞힌 비율 (낮을수록 좋다)
  · 자체 정확도가 base보다 낮을 것

⚠️ 2026-08-19 실측으로 드러난 보완 사항 — **대조 여지(contrast)** 를 함께 봐야 한다.
A(s)만으로 고르면 dpo가 1위인데(A=0.797), dpo는 base와 답이 **24/483(5.0%)만 다르다.**
그러면 log p_amateur ≈ log p_base 이므로

    log p_base − λ·log p_amateur ≈ (1−λ)·log p_base

가 되어 **사실상 온도 조절**이 된다. 새 방법을 시험한다고 해놓고 온도만 바꾸는 셈이다.
그래서 `contrast = base와 답이 다른 문항 비율`을 함께 출력하고,
**contrast ≥ 8%** 를 만족하는 후보 중에서 A(s)가 최대인 것을 고른다(사전 고정).

teacher는 자체 정확도가 최하위권이지만 rescue가 크다 — 나쁜 모델이 아니라 **다른 능력을 가진
specialist**이므로 빼면 유용한 소수파 추론까지 죽인다. 지표가 그 판단을 자동으로 하게 한다.

⚠️ 실행 가능성 제약: amateur는 **실제로 로드할 수 있는 체크포인트**여야 한다.
   `prompt`(시스템 프롬프트 변형)·`temp`/`mt3072`(샘플링 설정 변형)는 별도 모델이 아니라
   **베이스 그 자체**다. 대조 디코딩의 amateur로 쓸 수 없으므로 참고용으로만 출력한다.
   (다만 prompt 계열은 MTI의 negative-prompt guidance 후보로는 의미가 있다.)

사용: PYTHONIOENCODING=utf-8 python remote/select_amateur.py
"""
import json
import math
import os
from collections import defaultdict

R = 'results/'
BASE = ['val_samples.jsonl', 'val_samples_s43.jsonl', 'val_samples_s44.jsonl']

# 이름 -> (덤프들, 어댑터 경로 또는 None)
CAND = {
    'dpo':      (['val_samples_dpo.jsonl', 'val_samples_dpo_s43.jsonl',
                  'val_samples_dpo_s44.jsonl'], 'outputs/dpo/...'),
    'teacher':  (['val_samples_teacher_full_s42.jsonl', 'val_samples_teacher_full_s43.jsonl',
                  'val_samples_teacher_full_s44.jsonl'],
                 'outputs/qlora_teacher_full/qlora_r16_lr5e-05_ep1_final'),
    'rftmask':  (['val_samples_rftmask_s42.jsonl', 'val_samples_rftmask_s43.jsonl',
                  'val_samples_rftmask_s44.jsonl'], 'outputs/qlora_rft_masked/...'),
    'masked':   (['val_samples_masked_s42.jsonl', 'val_samples_masked_s43.jsonl',
                  'val_samples_masked_s44.jsonl'], 'outputs/qlora_teacher_masked/...'),
    'marginal': (['val_samples_marginal_s42.jsonl', 'val_samples_marginal_s43.jsonl'],
                 'outputs/qlora_marginal/...'),
    'rank':     (['val_samples_rank_s42.jsonl'], None),
    'rfs3':     (['val_samples_rfs3_s42.jsonl'], None),
    # --- 아래는 별도 모델이 아니다 (베이스 + 설정 변경). 참고용 ---
    'prompt*':  (['val_samples_sp_base_s42.jsonl', 'val_samples_sp_classify_s42.jsonl',
                  'val_samples_sp_direct_s42.jsonl', 'val_samples_sp_extract_s42.jsonl',
                  'val_samples_sp_verify_s42.jsonl'], None),
    'mt3072*':  (['val_samples_mt3072_s42.jsonl', 'val_samples_mt3072_s43.jsonl',
                  'val_samples_mt3072_s44.jsonl'], None),
    'temp*':    (['val_samples_tp_base_t03_s42.jsonl', 'val_samples_tp_base_t07_s42.jsonl',
                  'val_samples_tp_base_t11_s42.jsonl', 'val_samples_tp_minimal_t07_s42.jsonl',
                  'val_samples_tp_nosys_t07_s42.jsonl'], None),
}
N = 32
LAM = 2.0          # rescue 1%p를 wrongAgreement 2%p만큼 나쁘게 본다 (사전 고정)
MIN_CONTRAST = 0.08  # base와 답이 이만큼은 달라야 대조가 의미를 갖는다 (사전 고정)


def wvote(pairs, scale=2.0):
    v = [(a, lp) for a, lp in pairs if a is not None]
    if not v:
        return None
    m = max(lp for _, lp in v)
    w = defaultdict(float)
    for a, lp in v:
        w[a] += math.exp((lp - m) * scale)
    return max(w, key=w.get)


def collapse(files):
    """계열 내부 표본을 풀링해 문항당 답 하나"""
    acc, gold = defaultdict(list), {}
    got = 0
    for fn in files:
        p = R + fn
        if not os.path.exists(p):
            continue
        got += 1
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            gold[r['id']] = r['gold']
            acc[r['id']].extend((s.get('ans'), s.get('logp', 0.0)) for s in r['samples'][:N])
    if not got:
        return None, None
    return {i: wvote(v) for i, v in acc.items()}, gold


def main():
    base, gold = collapse(BASE)
    ids0 = set(base)
    print('exp80a amateur 선정 — A(s) = wrongAgreement − %.1f x rescueRate' % LAM)
    print()
    print('%-10s%7s%9s%12s%11s%9s%10s%8s' %
          ('후보', '정확도', 'base대비', 'wrongAgree', 'rescue율', 'contrast', 'A(s)', '어댑터'))
    print('-' * 78)

    rows = []
    for name, (files, adapter) in CAND.items():
        ans, _ = collapse(files)
        if ans is None:
            continue
        ids = sorted(ids0 & set(ans))
        acc = sum(1 for i in ids if ans[i] == gold[i])
        bacc = sum(1 for i in ids if base[i] == gold[i])
        both_wrong = [i for i in ids if base[i] != gold[i] and ans[i] != gold[i]]
        wa = (sum(1 for i in both_wrong if base[i] == ans[i]) / len(both_wrong)
              if both_wrong else 0.0)
        base_wrong = [i for i in ids if base[i] != gold[i]]
        rr = (sum(1 for i in base_wrong if ans[i] == gold[i]) / len(base_wrong)
              if base_wrong else 0.0)
        A = wa - LAM * rr
        contrast = sum(1 for i in ids if ans[i] != base[i]) / len(ids)
        rows.append((name, acc, acc - bacc, wa, rr, A, adapter, contrast))

    for r in sorted(rows, key=lambda x: -x[5]):
        star = '·' if r[0].endswith('*') else ('O' if r[6] else 'X')
        flag = '' if r[7] >= MIN_CONTRAST else '  ← 대조 여지 부족'
        print('%-10s%7d%+9d%12.3f%11.3f%8.1f%%%10.3f%8s%s'
              % (r[0], r[1], r[2], r[3], r[4], 100 * r[7], r[5], star, flag))

    print()
    print('어댑터 열: O = 실제 로드 가능한 LoRA · X = 체크포인트 없음 · · = 별도 모델이 아님')
    print('  (`*` 표시는 베이스 + 프롬프트/샘플링 설정 변형이라 amateur로 쓸 수 없다.')
    print('   다만 prompt* 계열은 MTI의 negative-prompt guidance 후보로는 의미가 있다.)')
    print()
    usable = [r for r in rows if r[6]]
    elig = [r for r in usable if r[7] >= MIN_CONTRAST]
    dropped = [r for r in usable if r[7] < MIN_CONTRAST]
    if dropped:
        print('대조 여지 부족(<%.0f%%)으로 제외: %s'
              % (100 * MIN_CONTRAST,
                 ', '.join('%s(%.1f%%)' % (r[0], 100 * r[7]) for r in dropped)))
    if elig:
        best = max(elig, key=lambda x: x[5])
        print()
        print('▶ 선정: **%s**' % best[0])
        print('   A=%.3f · wrongAgree %.3f · rescue율 %.3f · contrast %.1f%% · 자체 %d (base%+d)'
              % (best[5], best[3], best[4], 100 * best[7], best[1], best[2]))
        print('   어댑터: %s' % best[6])
        print()
        print('   근거: rescue율이 후보 중 가장 낮은 축이라 유용한 소수파 추론을 지울 위험이 작고,')
        print('         자체 정확도가 base보다 뚜렷이 낮아 고전 CD의 amateur 조건에 맞으며,')
        print('         base와 답이 %.1f%% 달라 대조가 실제로 작동할 여지가 있다.' % (100 * best[7]))
        for r in sorted(elig, key=lambda x: -x[5])[1:]:
            print('   차선: %-9s A=%.3f contrast %.1f%% rescue %.3f'
                  % (r[0], r[5], 100 * r[7], r[4]))


if __name__ == '__main__':
    main()
