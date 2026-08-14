#!/bin/bash
# 교사 풀이 생성 워커를 병렬로 띄운다 (멱등 — 이미 도는 offset은 건너뜀).
#
# 왜 병렬인가: `claude -p` 한 번은 순수한 대기 시간이라 CPU를 쓰지 않는다.
# 순차로 돌리면 1,200문제에 36시간이 걸리지만 CPU는 놀고 있었다(4코어에 load 1.0).
#
# 왜 배치가 작은가: 한 호출에 5문제를 넣었더니 호출당 5.7분이 걸렸고 일부는 900초
# 타임아웃에 걸렸다. 타임아웃 하나가 재시도까지 30분을 날린다(2026-08-14 실측:
# 워커 0은 10배치 중 4개만 성공). 배치를 2로 줄이면 호출이 짧아져 타임아웃이 사라진다.
#
# ⚠️ 스크립트 본문을 `bash -c "cat > x <<EOF ..."` 로 만들지 말 것 — 그 heredoc 내용이
# 만든 프로세스의 명령줄에 통째로 남아서 `pgrep -f` 가 자기 자신을 매칭한다
# (2026-08-13 PRM 체인이 이 함정으로 교착됐다). 반드시 파일로 두고 실행한다.
#
# 사용: bash remote/launch_teacher_workers.sh [워커수]
set -u
N=${1:-10}
TOTAL=1200
SPAN=$(( (TOTAL + N - 1) / N ))

export PATH=$HOME/.local/bin:$PATH
set -a; . $HOME/.ajudl_env; set +a
cd $HOME/work/deep-learning-challenge

for i in $(seq 0 $((N - 1))); do
  OFF=$((i * SPAN))
  if pgrep -f "offset $OFF " > /dev/null; then
    echo "offset $OFF — 이미 가동 중, 건너뜀"
    continue
  fi
  setsid nohup uv run python api/gen_teacher_claude_code.py \
      --src data/hard_problems_top.csv --offset $OFF --limit $SPAN \
      --batch 2 --out-dir data/teacher_out < /dev/null > teacher_w$i.log 2>&1 &
  echo "offset $OFF 시작 (담당 $SPAN문제)"
  sleep 1
done

sleep 5
echo "가동 워커: $(pgrep -cf gen_teacher_claude_code)"
