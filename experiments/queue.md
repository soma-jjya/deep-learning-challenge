# 실험 큐 (서버 러너가 위에서부터 순서대로 실행)

규칙: 로컬 Claude(계획자)가 실험을 추가하고, 서버 Claude(실행자)는 실행·기록·체크만 한다.
완료 표기: `- [x] ... → 결과: ...`

- [x] exp04-rft-gen: `remote/generate_rft.py` 실행해 data/sft.jsonl 생성. 완료 후 "정답 도달 문제 수 / 채택 풀이 수"를 기록. sft.jsonl은 커밋하지 말고(용량) 통계만 기록 → 결과: 16500/16500 처리, 정답 도달 문제 13190개(79.9%), 채택 풀이 36733개
- [x] exp04b-filter-sft: exp04가 오류 문항 제외 전에 시작됐으므로 사후 필터링. `deep-learning-challenge-2026/train_filtered_ids.csv`의 id에 해당하는 라인을 data/sft.jsonl에서 제거(각 라인의 "id" 필드 기준). 제거 전/후 라인 수를 기록 → 결과: 36733줄 → 36144줄 (589줄 제거)
- [x] exp05-eval-base: `remote/eval_vllm.py --mode both` 실행 (베이스 모델, greedy + SC n=8). 스크립트가 검증 500에서 오류 문항을 자동 제외함 — 출력되는 유효 검증 문항 수를 함께 기록. exp02·03에 해당하는 AWS 측 기준점 확보 → 결과: 유효 검증 483문항, greedy 69.4%(335/483), SC n=8 74.7%(361/483)
- [x] exp06-qlora-r16: SFT 학습 완료(loss 0.166, wandb o5zlvcyp), 평가 완료 → 결과: greedy 68.3%, SC n8 73.5% — **베이스(69.4/74.7)보다 하락**. 기록은 로컬 Claude가 대행 (러너 중단 사고로 서버 기록 생략됨)
- [x] exp06b-submission: 리더보드 제출 파일 생성. exp06 어댑터가 베이스보다 나쁘므로 **어댑터 없이** `remote/make_submission.py --n 8 --tag base` 실행 → results/submission_base.csv (831행·정수만 확인, **커밋 대상**) → 결과: results/submission_base.csv 생성 완료 (831행, 정수 답만, 결측 없음)
- [ ] exp07-sc-scale: **베이스 모델로** SC 스케일 실험 (H11). `remote/eval_vllm.py --mode sc --n 16` 실행, 이어서 `--n 4`도. exp05의 n=8(74.7%)과 함께 n별 정확도 표 기록 (제출용 n 선택 근거)
- [ ] exp06c-qlora-gentle: H3 재시도 — 1차 실패 원인 가설(과한 학습률·자기증류 과적합) 대응 (H14 결합). ① data/sft.jsonl에서 문제당 '가장 짧은' 정답 풀이 1개만 남긴 data/sft_short.jsonl 생성(파이썬으로, id별 min len) ② remote/train_qlora.py CONFIG를 lr=5e-5, epochs=1, data_path=data/sft_short.jsonl로 수정 후 학습 ③ `remote/eval_vllm.py --mode both --adapter <새 어댑터>` 평가·기록. 기존 어댑터 디렉토리는 삭제하지 말 것
- [ ] exp08-rft-round2: 반복 RFT (H10) — exp06c 어댑터가 베이스보다 좋을 때만 실행, 아니면 skip으로 기록. `remote/generate_rft.py`를 exp06c 어댑터로(vLLM LoRA 로드 추가 필요 — eval_vllm.py 참고) 출력 data/sft_round2.jsonl. 정답 도달률을 exp04(79.9%)와 비교
