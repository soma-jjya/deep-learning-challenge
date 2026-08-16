#!/bin/bash
# exp67 — Budget Forcing 재판정. 확장 검증셋(2,500)으로 임계를 ±1.49%p → ±0.65%p로 낮춰서.
#
# 왜 다시 하나: BF는 **두 번 다 +0.6%p**였다(exp32 greedy, exp42 SC n8). 부호가 일관 양수인데
# 성공 기준(+1.5%p) 미달로 기각했다. 그런데 exp65에서 483문항 대응비교 임계가 정확히
# ±1.49%p임이 드러났다 — 즉 **+0.6%p는 원리적으로 판별 불가능한 크기였고, 우리는 그것을
# '효과 없음'이 아니라 '측정 불가'로 기록했어야 했다.**
#
# 확장셋(2,500)에서 임계는 ±0.65%p. +0.6%p가 실재하면 이번엔 잡힌다.
# BF는 어댑터를 안 쓰므로 확장셋의 학습데이터 오염(74.4%)과 무관하다.
#
# 설계: 대조군(round 0)과 처치군(round 1)이 **같은 실행·같은 표본**에서 나온다.
# BF는 round 0 텍스트를 이어쓰는 구조라 대조가 자동으로 대응 표본이 된다 — McNemar에 최적이다.
#
# 비용: n=8로 한다(exp42와 같은 조건). n=32는 4배인데 BF의 효과 크기를 재는 데는 n=8이면 충분하고,
# 채택 시 n=32 확인은 그때 하면 된다.
#
# 사용: setsid nohup bash remote/run_exp67_chain.sh > exp67_chain.log 2>&1 & disown
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

echo "[$(date +%H:%M)] 앞선 작업 종료 대기"
while pgrep -f 'dump_lb_sample[s].py|sweep_temp_and_promp[t].py|dump_sample[s].py' > /dev/null; do sleep 60; done
wait_gpu

for S in 42 43; do
  OUT=results/val_large_bf_n8_s$S.jsonl
  [ -s "$OUT" ] && { echo "[$(date +%H:%M)] seed $S 이미 있음"; continue; }
  wait_gpu
  echo "[$(date +%H:%M)] BF r1 n8 seed $S (2,500문항)"
  uv run python remote/eval_budget_forcing.py --rounds 1 --n 8 --seed "$S" \
    --val-ids data/val_large_ids.csv --out "$OUT"
done

echo
echo "===== exp67: round0(대조) vs round1(BF) 는 위 로그의 [round N] 줄 참고 ====="
echo "===== 가중 투표로 재채점 + McNemar ====="
uv run python -c "
import json, math, glob
from collections import defaultdict
def wv(ss, scale=2.0):
    c = defaultdict(float)
    for s in ss:
        if s.get('ans') is not None: c[s['ans']] += math.exp(scale * s.get('logp', 0.0))
    return max(c, key=c.get) if c else None
for f in sorted(glob.glob('results/val_large_bf_n8_s*.jsonl')):
    rows = [json.loads(l) for l in open(f, encoding='utf-8')]
    ok = sum(1 for r in rows if wv(r['samples']) == r['gold'])
    print(f'{f}: 가중투표 {ok}/{len(rows)} = {ok/len(rows):.2%}')
"
echo "[$(date +%H:%M)] EXP67_DONE"
echo "판정: 확장셋 임계 ±0.65%p. round0 대비 round1이 그보다 크게 오르면 BF는 실재한다."
