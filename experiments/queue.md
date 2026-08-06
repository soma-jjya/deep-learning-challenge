# 실험 큐 (서버 러너가 위에서부터 순서대로 실행)

규칙: 로컬 Claude(계획자)가 실험을 추가하고, 서버 Claude(실행자)는 실행·기록·체크만 한다.
완료 표기: `- [x] ... → 결과: ...`

- [x] exp04-rft-gen: `remote/generate_rft.py` 실행해 data/sft.jsonl 생성. 완료 후 "정답 도달 문제 수 / 채택 풀이 수"를 기록. sft.jsonl은 커밋하지 말고(용량) 통계만 기록 → 결과: 16500/16500 처리, 정답 도달 문제 13190개(79.9%), 채택 풀이 36733개
- [x] exp04b-filter-sft: exp04가 오류 문항 제외 전에 시작됐으므로 사후 필터링. `deep-learning-challenge-2026/train_filtered_ids.csv`의 id에 해당하는 라인을 data/sft.jsonl에서 제거(각 라인의 "id" 필드 기준). 제거 전/후 라인 수를 기록 → 결과: 36733줄 → 36144줄 (589줄 제거)
- [x] exp05-eval-base: `remote/eval_vllm.py --mode both` 실행 (베이스 모델, greedy + SC n=8). 스크립트가 검증 500에서 오류 문항을 자동 제외함 — 출력되는 유효 검증 문항 수를 함께 기록. exp02·03에 해당하는 AWS 측 기준점 확보 → 결과: 유효 검증 483문항, greedy 69.4%(335/483), SC n=8 74.7%(361/483)
- [x] exp06-qlora-r16: SFT 학습 완료(loss 0.166, wandb o5zlvcyp), 평가 완료 → 결과: greedy 68.3%, SC n8 73.5% — **베이스(69.4/74.7)보다 하락**. 기록은 로컬 Claude가 대행 (러너 중단 사고로 서버 기록 생략됨)
- [x] exp06b-submission: 리더보드 제출 파일 생성. exp06 어댑터가 베이스보다 나쁘므로 **어댑터 없이** `remote/make_submission.py --n 8 --tag base` 실행 → results/submission_base.csv (831행·정수만 확인, **커밋 대상**) → 결과: results/submission_base.csv 생성 완료 (831행, 정수 답만, 결측 없음)
- [x] exp07-sc-scale: **베이스 모델로** SC 스케일 실험 (H11). `remote/eval_vllm.py --mode sc --n 16` 실행, 이어서 `--n 4`도. exp05의 n=8(74.7%)과 함께 n별 정확도 표 기록 (제출용 n 선택 근거) → 결과: n=4 72.7%(351/483), n=8 74.7%(361/483, exp05), n=16 75.6%(365/483). n=8→16은 +0.9%p로 체감 — 제출은 비용 대비 n=8~16 사이 선택
- [x] exp06c-qlora-gentle: H3 재시도 (lr 5e-5, ep1, 최단풀이). 어댑터: `outputs/qlora_gentle/qlora_r16_lr5e-05_ep1_final`, train loss 0.2429, wandb sv8lkhho. `remote/eval_vllm.py --mode both --adapter outputs/qlora_gentle/qlora_r16_lr5e-05_ep1_final` 실행 완료(eval06c.log, results/eval_outputs_qlora_gentle_qlora_r16_lr5e-05_ep1_final.json) → 결과: 검증 483문항, greedy 67.9%(328/483), SC n8 73.3%(354/483). exp05 베이스(69.4/74.7) 대비 greedy −1.5%p, SC −1.4%p로 **하락** — exp06에 이어 QLoRA SFT 2연속 실패. 정체 카운트 2/3
- [x] exp08-rft-round2: 반복 RFT (H10) — exp06c 어댑터가 베이스보다 좋을 때만 실행, 아니면 skip으로 기록 → 결과: **skip**. exp06c도 베이스보다 하락(-1.5%p greedy, -1.4%p SC)해 H10 전제(개선된 모델로 재생성) 불성립. `remote/generate_rft.py` 미실행
