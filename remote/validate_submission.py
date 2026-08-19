"""제출 파일 전수 검증 — 8/31 당일 제출 직전에 반드시 통과시킨다. GPU·pandas 불필요.

docs/final-pipeline.md 3단계의 체크리스트를 **사람 눈이 아니라 코드**로 옮긴 것이다.
2026-08-15 사고가 근거다 — 20문항만 덤프한 상태로 만든 831행 파일이 행수·결측·중복 검사를
전부 통과했다. 눈으로 보는 검증은 이미 한 번 실패했다.

검사 항목
  1. 헤더가 정확히 id,answer 이고 **id가 소문자**인가 (2026-07-31 첫 제출 ERROR의 원인)
  2. 행 수 == test 파일 문항 수, id 집합이 정확히 일치, 중복·결측 0
  3. answer가 전부 정수이고 **int64 범위 안**인가 (부호·자릿수 포함)
  4. answer가 0인 문항 수 — 덤프 누락의 지표
  5. (덤프를 함께 주면) 덤프 커버리지·문항당 표본 수·잘림/추출실패 비율

사용:
    python remote/validate_submission.py --sub results/submission_final.csv \
        --test deep-learning-challenge-2026/<최종test>.csv [--dump results/final_samples.jsonl --n 32]
"""
import argparse
import csv
import json
import os
import sys

csv.field_size_limit(10 ** 7)
INT64_MAX = 2 ** 63 - 1

FAIL, WARN = [], []


def bad(m):
    FAIL.append(m)
    print('  ⛔ ' + m)


def warn(m):
    WARN.append(m)
    print('  ⚠️ ' + m)


def ok(m):
    print('  ✅ ' + m)


def read_csv_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        r = csv.reader(f)
        rows = list(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sub', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--dump', default=None)
    ap.add_argument('--n', type=int, default=None)
    args = ap.parse_args()

    print('제출 파일 검증: %s' % args.sub)
    print()

    # ── 1. 헤더 ──
    print('[1] 헤더')
    rows = read_csv_rows(args.sub)
    if not rows:
        bad('파일이 비어 있다')
        return finish()
    head = rows[0]
    if head == ['id', 'answer']:
        ok('헤더 id,answer (소문자)')
    else:
        bad('헤더가 %r 이다. 정확히 [\'id\',\'answer\'] 여야 한다 '
            '(대회 문서엔 ID로 적혀 있으나 채점기는 소문자)' % head)
    body = rows[1:]

    # ── 2. 문항 집합 ──
    print('[2] 문항 집합')
    trows = read_csv_rows(args.test)
    th = [c.strip() for c in trows[0]]
    tid_col = next((i for i, c in enumerate(th) if c.lower() == 'id'), None)
    if tid_col is None:
        bad('test 파일에서 id 컬럼을 못 찾았다: %s' % th)
        return finish()
    test_ids = [r[tid_col] for r in trows[1:] if r]
    sub_ids = [r[0] for r in body if r]
    if len(sub_ids) == len(test_ids):
        ok('행 수 %d == test 문항 수 %d' % (len(sub_ids), len(test_ids)))
    else:
        bad('행 수 %d != test 문항 수 %d' % (len(sub_ids), len(test_ids)))
    if len(set(sub_ids)) != len(sub_ids):
        dup = len(sub_ids) - len(set(sub_ids))
        bad('제출 파일에 중복 id %d건' % dup)
    else:
        ok('중복 id 없음')
    miss = set(test_ids) - set(sub_ids)
    extra = set(sub_ids) - set(test_ids)
    if miss:
        bad('test에는 있는데 제출에 없는 id %d개 (예: %s)'
            % (len(miss), ', '.join(list(miss)[:3])))
    if extra:
        bad('제출에만 있는 id %d개 (예: %s)' % (len(extra), ', '.join(list(extra)[:3])))
    if not miss and not extra:
        ok('id 집합 완전 일치')

    # ── 3. 답 형식 ──
    print('[3] answer 값')
    nonint, oversize, blanks, vals = [], [], [], []
    for r in body:
        if len(r) < 2 or r[1] == '':
            blanks.append(r[0] if r else '?')
            continue
        v = r[1].strip()
        try:
            iv = int(v)
        except ValueError:
            nonint.append((r[0], v[:30]))
            continue
        vals.append(iv)
        if abs(iv) > INT64_MAX:
            oversize.append((r[0], len(v)))
    if blanks:
        bad('answer가 빈 문항 %d개 (예: %s)' % (len(blanks), ', '.join(blanks[:3])))
    if nonint:
        bad('정수가 아닌 answer %d개 (예: %s)'
            % (len(nonint), ', '.join('%s=%s' % t for t in nonint[:3])))
    if oversize:
        bad('int64 범위를 벗어난 answer %d개 (예: %s) — 채점기에서 깨질 수 있다'
            % (len(oversize), ', '.join('%s(%d자리)' % t for t in oversize[:3])))
    if not (blanks or nonint or oversize):
        ok('전부 int64 범위 안의 정수 (%d개)' % len(vals))
    if vals:
        # 임계값은 추측이 아니라 우리 실측으로 잡았다 (2026-08-19, 제출 파일 25건 전수):
        #   · 정상 제출 24건의 0 비율 = 2.05% ~ 3.25% (중앙 2.77%)
        #   · 2026-08-15 사고 파일(submission_smokefinal) = 97.71%
        #   · 학습셋 gold 자체도 1.31%가 답이 0이다 — 0은 정당한 답이다
        # 즉 정상과 사고 사이가 크게 벌어져 있으므로 4.5%/8%로 두면 양쪽에 충분한 여유가 있다.
        # (덤프를 --dump로 함께 주면 누락은 id 집합 대조로 정확히 잡히고, 이 검사는 그 보조망이다.)
        nz = sum(1 for v in vals if v == 0)
        neg = sum(1 for v in vals if v < 0)
        pct = 100.0 * nz / len(vals)
        line = '0인 답 %d개 (%.2f%%) · 음수 %d개 · 최대 자릿수 %d' % (
            nz, pct, neg, max(len(str(abs(v))) for v in vals))
        if pct > 8.0:
            bad(line + '  ← 0이 비정상적으로 많다(정상 2.0~3.3%). 덤프 누락을 의심할 것')
        elif pct > 4.5:
            warn(line + '  ← 정상 범위(2.0~3.3%)보다 높다. 덤프 커버리지를 확인할 것')
        else:
            ok(line)

    # ── 4. 덤프 대조 (선택) ──
    if args.dump:
        print('[4] 덤프 대조')
        if not os.path.exists(args.dump):
            bad('덤프 파일이 없다: %s' % args.dump)
        else:
            seen, counts, tot, none, trunc = set(), [], 0, 0, 0
            dup = 0
            for line in open(args.dump, encoding='utf-8'):
                d = json.loads(line)
                if d['id'] in seen:
                    dup += 1
                seen.add(d['id'])
                counts.append(len(d['samples']))
                for s in d['samples']:
                    tot += 1
                    none += s.get('ans') is None
                    trunc += bool(s.get('trunc'))
            uncovered = set(test_ids) - seen
            if uncovered:
                bad('덤프가 %d문항을 덮지 않는다 — 그 문항은 0으로 채워졌다 (예: %s)'
                    % (len(uncovered), ', '.join(list(uncovered)[:3])))
            else:
                ok('덤프가 test 전 문항을 덮는다 (%d문항)' % len(seen))
            if dup:
                warn('덤프에 중복 id %d건 (재개 중 사망 흔적 — 뒤쪽 값이 사용됨)' % dup)
            if counts:
                lo, hi = min(counts), max(counts)
                if args.n and lo < args.n:
                    bad('문항당 표본 수가 요청 n=%d에 못 미침 (최소 %d) — 덤프가 덜 찼다'
                        % (args.n, lo))
                elif lo != hi:
                    warn('문항당 표본 수가 균일하지 않다 (%d~%d)' % (lo, hi))
                else:
                    ok('문항당 표본 수 %d개로 균일' % lo)
            if tot:
                ok('표본 %d개 · 답 추출 실패 %d (%.3f%%) · 잘림 %d (%.2f%%)'
                   % (tot, none, 100.0 * none / tot, trunc, 100.0 * trunc / tot))
                if 100.0 * none / tot > 1.0:
                    bad('추출 실패율이 1%%를 넘는다 — 형식 붕괴 의심 (평소 0.001%%)')
                if 100.0 * trunc / tot > 5.0:
                    warn('잘림 비율이 5%%를 넘는다 (평소 1.7~1.9%%) — max_tokens 확인')
    return finish()


def finish():
    print()
    if FAIL:
        print('⛔ 실패 %d건 — 이 파일을 제출하면 안 된다' % len(FAIL))
        for m in FAIL:
            print('   · ' + m)
        sys.exit(1)
    if WARN:
        print('⚠️ 경고 %d건 — 내용을 확인하고 판단할 것' % len(WARN))
        for m in WARN:
            print('   · ' + m)
    print('✅ 검증 통과')
    return 0


if __name__ == '__main__':
    main()
