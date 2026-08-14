#!/bin/bash
# exp53 평가 체인 — 교사 증류 어댑터를 시드 42/43/44로 덤프하고 쌍별로 비교한다.
#
# 왜 시드 3개인가: exp47이 같은 어댑터를 두 번 재서 1.4%p가 어긋났다. 단일 측정으로는
# 실효와 노이즈를 구분할 수 없다는 것이 exp48의 결론이고, 그래서 판정은 **부호 일관 +
# 평균 +1.5%p 이상**으로 못박았다. 베이스 덤프(val_samples*.jsonl)는 이미 있으므로
# 어댑터 쪽만 새로 뜬다.
#
# ⚠️ 이 파일을 `bash -c "cat > x <<EOF ..."` 로 만들지 말 것 — heredoc 본문이 만든
# 프로세스의 명령줄에 남아 `pgrep -f`가 자기 자신을 매칭한다(2026-08-13 PRM 체인 교착).
#
# 사용: setsid nohup bash remote/run_teacher_eval_chain.sh <어댑터경로> > eval53.log 2>&1 & disown
set -u
ADAPTER=${1:?어댑터 경로가 필요합니다}
export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

for S in 42 43 44; do
  OUT=results/val_samples_teacher_s$S.jsonl
  if [ -s "$OUT" ]; then
    echo "[$(date +%H:%M)] seed $S — 이미 있음, 건너뜀"
    continue
  fi
  echo "[$(date +%H:%M)] seed $S 덤프 시작"
  uv run python remote/dump_samples.py --n 32 --seed "$S" --adapter "$ADAPTER" --out "$OUT"
done

echo
echo "===== 시드별 가중 투표 정확도 ====="
for S in 42 43 44; do
  case $S in
    42) BASE=results/val_samples.jsonl ;;
    *)  BASE=results/val_samples_s$S.jsonl ;;
  esac
  echo "--- seed $S : 베이스 ($BASE)"
  uv run python remote/analyze_selection_gap.py --samples "$BASE" --n 32 | head -6
  echo "--- seed $S : 교사 어댑터"
  uv run python remote/analyze_selection_gap.py --samples "results/val_samples_teacher_s$S.jsonl" --n 32 | head -6
done

echo
echo "⚠️ 판정: 3시드 부호 일관 + 평균 +1.5%p 이상일 때만 채택. 최종 스택 변경은 사용자 승인 사항."
