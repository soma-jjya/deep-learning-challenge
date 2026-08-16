#!/bin/bash
# exp66 — 확장 검증셋(2,500)의 베이스 기준선. 남은 모든 판정이 쓸 새 자.
#
# 왜: 483문항 대응비교 임계가 ±1.49%p인데 성공 기준이 정확히 +1.5%p였다(exp65).
# 임계선에 걸쳐 판정해왔고 여유가 0이었다. 2,500이면 ±0.65%p가 되어,
# 부호가 일관 양수였으나 기각한 것들(Budget Forcing +0.6%p, 검증자 +0.4~0.5%p)이
# 판별 가능해진다. 합치면 +1.1%p인데 지금 자로는 그것도 못 잰다.
#
# ⚠️ 확장셋의 74.4%가 우리 학습 데이터 생성에 쓰였다. **베이스 평가에는 무관**하지만
#    (베이스는 그 데이터를 본 적 없다) 어댑터 평가에는 오염이다. 어댑터를 잴 때는
#    data/val_large_ids.csv의 used_in_training_data == False 인 641문항만 써야 한다.
#
# 사용: setsid nohup bash remote/run_exp66_chain.sh > exp66_chain.log 2>&1 & disown
set -u
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

wait_gpu() {
  for i in $(seq 1 720); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "${FREE:-0}" -gt 18000 ] && [ "$APPS" -eq 0 ]; then return 0; fi
    sleep 30
  done
}

echo "[$(date +%H:%M)] 앞선 작업(exp64) 종료 대기"
while pgrep -f 'dump_lb_sample[s].py|sweep_temp_and_promp[t].py' > /dev/null; do sleep 60; done
wait_gpu

for S in 42 43; do
  OUT=results/val_large_s$S.jsonl
  [ -s "$OUT" ] && { echo "[$(date +%H:%M)] seed $S 이미 있음"; continue; }
  wait_gpu
  echo "[$(date +%H:%M)] 확장셋 베이스 덤프 seed $S (2,500문항 × 32)"
  uv run python remote/dump_samples.py --n 32 --seed "$S" \
    --val-ids data/val_large_ids.csv --out "$OUT"
done

echo
echo "===== 확장셋 기준선 ====="
for S in 42 43; do
  printf 'seed %s  ' "$S"
  uv run python remote/analyze_selection_gap.py --samples "results/val_large_s$S.jsonl" --n 32 | sed -n '2p'
done
echo
echo "===== 기존 483문항 부분집합으로 재확인 (연속성) ====="
uv run python -c "
import json, math, csv
from collections import defaultdict
old = {r['id'] for r in csv.DictReader(open('data/val_large_ids.csv', encoding='utf-8')) if r['in_old_val'] == 'True'}
def wv(ss, scale=2.0):
    c = defaultdict(float)
    for s in ss:
        if s.get('ans') is not None: c[s['ans']] += math.exp(scale * s.get('logp', 0.0))
    return max(c, key=c.get) if c else None
for S in (42, 43):
    rows = [json.loads(l) for l in open(f'results/val_large_s{S}.jsonl', encoding='utf-8')]
    sub = [r for r in rows if r['id'] in old]
    ok = sum(1 for r in sub if wv(r['samples'][:32]) == r['gold'])
    print(f'  seed {S}: 기존 483 부분집합 {ok}/{len(sub)} = {ok/max(1,len(sub)):.1%}  (과거 366/483=75.8% 대역이어야 정상)')
"
echo "[$(date +%H:%M)] EXP66_DONE"
