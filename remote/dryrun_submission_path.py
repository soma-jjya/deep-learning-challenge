"""8/31 제출 경로 dry-run — 모델 추론 없이 병리 케이스를 주입해 가드가 실제로 작동하는지 본다.

왜 필요한가: 제출 경로의 사고는 전부 **드문 입력**에서 났다.
  · 2026-08-15 — 덤프가 20문항만 있는데 831행 파일이 검증을 전부 통과했다
  · 2026-08-19 — int64를 넘는 답이 투표에서 이기는 사례가 실측으로 확인됐다(n=32에서 6건)
이런 입력은 정상 실행에서는 몇 백 문항에 한 번 나오므로, **일부러 만들어 넣어야** 검증된다.
GPU도 vLLM도 pandas도 쓰지 않는다.

사용: PYTHONIOENCODING=utf-8 python remote/dryrun_submission_path.py
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_submission_from_dump import RULES, apply_rule_safe, INT64_MAX  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print('  %s %s%s' % ('✅' if cond else '⛔', name, ('  — ' + detail) if detail else ''))


def S(ans, logp=-0.1, trunc=False, ln=400):
    return {'ans': ans, 'logp': logp, 'trunc': trunc, 'len': ln}


def main():
    print('=== A. int64 초과 답 가드 ===')
    huge = int('6' * 2048)
    # 거대한 답이 다수를 차지하는 최악의 경우
    ss = [S(huge)] * 5 + [S(42)] * 3
    for rule in ('majority', 'weighted', 'trim25'):
        a, why = apply_rule_safe(RULES[rule], ss)
        check('%s: 거대 답이 다수여도 범위 안 후보로 대체' % rule,
              a == 42 and why and why.startswith('oversized'), 'a=%s why=%s' % (a, why))
    # 모든 답이 거대한 경우 → 0으로 두되 사유를 남긴다
    a, why = apply_rule_safe(RULES['weighted'], [S(huge)] * 4)
    check('전부 거대하면 0으로 두고 사유 기록', a == 0 and why == 'all_oversized',
          'a=%s why=%s' % (a, why))
    # 정상 입력은 건드리지 않는다
    a, why = apply_rule_safe(RULES['weighted'], [S(7)] * 3 + [S(9)])
    check('정상 입력은 결과·사유 모두 그대로', a == 7 and why is None, 'a=%s why=%s' % (a, why))
    # 경계값
    a, _ = apply_rule_safe(RULES['majority'], [S(INT64_MAX)] * 3)
    check('int64 최댓값은 통과', a == INT64_MAX)
    a, why = apply_rule_safe(RULES['majority'], [S(INT64_MAX + 1)] * 3 + [S(5)])
    check('int64 최댓값+1은 차단', a == 5 and why.startswith('oversized'))
    # 답이 전부 None
    a, why = apply_rule_safe(RULES['weighted'], [S(None)] * 4)
    check('답이 전부 None이면 0', a == 0, 'a=%s why=%s' % (a, why))

    print()
    print('=== B. 검증기가 병리 제출 파일을 실제로 잡는가 ===')
    tmp = tempfile.mkdtemp(prefix='ajudl_dryrun_')
    test_csv = os.path.join(tmp, 'test.csv')
    with open(test_csv, 'w', encoding='utf-8', newline='') as f:
        f.write('id,question' + chr(10))
        for i in range(20):
            f.write('t-%03d,dummy question %d' % (i, i) + chr(10))

    def write_sub(name, lines, header='id,answer'):
        p = os.path.join(tmp, name)
        with open(p, 'w', encoding='utf-8', newline='') as f:
            f.write(header + chr(10))
            for l in lines:
                f.write(l + chr(10))
        return p

    def run(sub, dump=None, n=None):
        cmd = [sys.executable, 'remote/validate_submission.py', '--sub', sub, '--test', test_csv]
        if dump:
            cmd += ['--dump', dump]
        if n:
            cmd += ['--n', str(n)]
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           encoding='utf-8', errors='replace')
        return r.returncode, (r.stdout or '') + (r.stderr or '')

    good = ['t-%03d,%d' % (i, i + 1) for i in range(20)]
    rc, out = run(write_sub('good.csv', good))
    check('정상 파일은 통과', rc == 0, 'rc=%d' % rc)

    rc, out = run(write_sub('upper.csv', good, header='ID,answer'))
    check('대문자 ID 헤더를 잡는다', rc == 1 and '소문자' in out)

    rc, out = run(write_sub('short.csv', good[:15]))
    check('행 수 부족을 잡는다', rc == 1 and '행 수' in out)

    bad_ids = ['x-%03d,%d' % (i, i) for i in range(20)]
    rc, out = run(write_sub('ids.csv', bad_ids))
    check('id 불일치를 잡는다', rc == 1 and ('없는 id' in out or '제출에만' in out))

    zeros = ['t-%03d,0' % i for i in range(19)] + ['t-019,7']
    rc, out = run(write_sub('zeros.csv', zeros))
    check('0이 대부분인 파일을 잡는다 (2026-08-15 사고 재현)',
          rc == 1 and '0이 비정상적으로 많다' in out)

    ov = ['t-%03d,%d' % (i, i + 1) for i in range(19)] + ['t-019,%d' % (INT64_MAX + 10)]
    rc, out = run(write_sub('over.csv', ov))
    check('int64 초과 값을 잡는다', rc == 1 and 'int64' in out)

    dupl = ['t-%03d,%d' % (i, i + 1) for i in range(19)] + ['t-000,5']
    rc, out = run(write_sub('dup.csv', dupl))
    check('중복 id를 잡는다', rc == 1 and '중복' in out)

    print()
    print('=== C. 덤프 커버리지 검사 ===')
    dump = os.path.join(tmp, 'partial.jsonl')
    with open(dump, 'w', encoding='utf-8') as f:
        for i in range(5):          # 20문항 중 5개만
            f.write(json.dumps({'id': 't-%03d' % i,
                                'samples': [S(i + 1) for _ in range(32)]}) + chr(10))
    rc, out = run(write_sub('good2.csv', good), dump=dump, n=32)
    check('부분 덤프를 잡는다', rc == 1 and '덮지 않는다' in out)

    dump2 = os.path.join(tmp, 'thin.jsonl')
    with open(dump2, 'w', encoding='utf-8') as f:
        for i in range(20):
            k = 8 if i == 3 else 32     # 한 문항만 표본 부족
            f.write(json.dumps({'id': 't-%03d' % i,
                                'samples': [S(i + 1) for _ in range(k)]}) + chr(10))
    rc, out = run(write_sub('good3.csv', good), dump=dump2, n=32)
    check('문항당 표본 부족을 잡는다', rc == 1 and '못 미침' in out)

    print()
    print('통과 %d / 실패 %d' % (len(PASS), len(FAIL)))
    if FAIL:
        print('실패 항목: ' + ', '.join(FAIL))
        sys.exit(1)
    print('✅ 제출 경로 dry-run 전부 통과 (GPU·vLLM·pandas 미사용)')


if __name__ == '__main__':
    main()
