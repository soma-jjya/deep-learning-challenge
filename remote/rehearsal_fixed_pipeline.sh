#!/bin/bash
# exp83 — 2026-08-19 감사에서 고친 제출 경로를 GPU로 실제 통과시키는 리허설.
#
# 왜 필요한가: 오늘 dump_lb_samples.py / make_submission_from_dump.py / make_submission.py를
# 고쳤는데 **GPU 경로로 한 번도 돌려보지 않았다.** exp30 리허설은 수정 전 코드 기준이다.
# 지금 상태는 "고쳤다고 믿는 코드"이고, 8/31에 처음 실행되는 코드가 되어서는 안 된다.
#
# 설계: 831문항 2.5시간을 다시 태우지 않는다. 생성 코드(vLLM 호출)는 바뀌지 않았고
# exp30이 이미 처리량을 실측했다. 바뀐 것은 **재개·지문·컬럼·가드 로직**이므로
# 작은 표본으로 그 경로들만 전부 밟는다 (약 10~15분).
#
# 사용: bash remote/rehearsal_fixed_pipeline.sh
set -u
export PATH=$HOME/.local/bin:$PATH
cd $HOME/work/deep-learning-challenge
LB=deep-learning-challenge-2026/deep_chal_math_leaderboard_filtered.csv
D=results/reh_samples.jsonl
PASS=0; FAIL=0
ok()  { echo "  [OK]   $1"; PASS=$((PASS+1)); }
ng()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "=== exp83 리허설: 수정된 제출 경로 전수 통과 ==="
rm -f "$D" "$D.progress" results/submission_reh.csv results/reh_test.csv

echo
echo "[1] 덤프 생성 (40문항 x n=8) — 파라미터 지문이 기록되는가"
uv run python remote/dump_lb_samples.py --n 8 --limit 40 --lb-csv "$LB" --out "$D" \
  > reh_1.log 2>&1
if [ "$(wc -l < "$D")" -eq 40 ]; then ok "40문항 생성됨"; else ng "행 수 $(wc -l < "$D")"; fi
if python3 -c "import json,sys; d=json.load(open('$D.progress')); sys.exit(0 if 'params' in d else 1)"; then
  ok "progress에 파라미터 지문 기록됨"
  python3 -c "import json; print('        지문:', json.load(open('$D.progress'))['params'])"
else ng "progress에 params 없음"; fi

echo
echo "[2] 다른 --n 으로 재개 시도 — 거부되어야 한다"
if uv run python remote/dump_lb_samples.py --n 16 --limit 40 --lb-csv "$LB" --out "$D" \
     > reh_2.log 2>&1; then
  ng "설정이 다른데 재개가 통과했다"
else
  if grep -q "이어서 진행하려는 설정이 이전 실행과 다르다" reh_2.log; then
    ok "설정 불일치를 잡아 거부함"
    grep -o "{'n':[^}]*}" reh_2.log | head -1 | sed 's/^/        /'
  else ng "거부는 됐으나 사유가 지문 불일치가 아니다"; fi
fi

echo
echo "[3] 같은 설정으로 재개 — 재생성 없이 즉시 완료되어야 한다"
uv run python remote/dump_lb_samples.py --n 8 --limit 40 --lb-csv "$LB" --out "$D" \
  > reh_3.log 2>&1
if [ "$(wc -l < "$D")" -eq 40 ] && grep -q "이어서 진행: 40문항 완료됨" reh_3.log; then
  ok "완료분을 건너뛰고 중복 없이 종료"
else ng "재개 후 행 수 $(wc -l < "$D")"; fi

echo
echo "[4] 청크 중간 사망 모사 — progress에서 5건을 지워도 재생성하지 않아야 한다"
python3 - <<'PY'
import json
p='results/reh_samples.jsonl.progress'
d=json.load(open(p)); d['done']=d['done'][:-5]; json.dump(d,open(p,'w'))
print('        progress의 done을 %d개로 줄임'%len(d['done']))
PY
uv run python remote/dump_lb_samples.py --n 8 --limit 40 --lb-csv "$LB" --out "$D" \
  > reh_4.log 2>&1
if grep -q "덤프에 이미 있는 문항 5개" reh_4.log && [ "$(wc -l < "$D")" -eq 40 ]; then
  ok "jsonl을 읽어 done을 자동 보정, 중복 기록 없음"
else ng "자동 보정 실패 (행 수 $(wc -l < "$D"))"; fi

echo
echo "[5] 제출 파일 생성 — 부분 덤프는 기본적으로 거부되어야 한다"
if uv run python remote/make_submission_from_dump.py --dump "$D" --rule weighted --n 8 \
     --lb-csv "$LB" --tag reh > reh_5.log 2>&1; then
  ng "40/831 덤프인데 제출 파일이 만들어졌다"
else
  grep -q "덤프가 791/831문항을 덮지 않는다" reh_5.log && ok "부분 덤프를 거부함" \
    || ng "거부 사유가 커버리지가 아니다"
fi

echo
echo "[6] 40문항짜리 test로 정상 경로 통과 + int64 가드 로그"
python3 - <<'PY'
import csv, json
ids=[json.loads(l)['id'] for l in open('results/reh_samples.jsonl',encoding='utf-8')]
src={r['id']:r for r in csv.DictReader(open(
    'deep-learning-challenge-2026/deep_chal_math_leaderboard_filtered.csv',encoding='utf-8'))}
w=csv.writer(open('results/reh_test.csv','w',newline='',encoding='utf-8'))
w.writerow(['id','question'])
for i in ids: w.writerow([i, src[i]['question']])
print('        40문항 test 파일 생성')
PY
uv run python remote/make_submission_from_dump.py --dump "$D" --rule weighted --n 8 \
  --lb-csv results/reh_test.csv --tag reh > reh_6.log 2>&1 \
  && ok "제출 파일 생성됨" || ng "생성 실패"
grep -E "^점검:|int64|0으로 둔" reh_6.log | sed 's/^/        /'

echo
echo "[7] 제출 전 검증기"
if uv run python remote/validate_submission.py --sub results/submission_reh.csv \
     --test results/reh_test.csv --dump "$D" --n 8 > reh_7.log 2>&1; then
  ok "검증 통과 (종료 코드 0)"
else
  ng "검증 실패"; grep -E "⛔" reh_7.log | head -5 | sed 's/^/        /'
fi

echo
echo "[8] make_submission.py 는 기본적으로 거부되어야 한다"
if uv run python remote/make_submission.py --n 8 --tag nope > reh_8.log 2>&1; then
  ng "체크포인트 없는 경로가 그냥 실행됐다"
else
  grep -q "중단 시 재개가 불가능하다" reh_8.log && ok "오사용 차단됨" || ng "다른 이유로 실패"
fi

echo
echo "=== 통과 $PASS / 실패 $FAIL ==="
[ "$FAIL" -eq 0 ] && echo "REHEARSAL_OK" || echo "REHEARSAL_FAILED"
