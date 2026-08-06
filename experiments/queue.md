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
- [x] exp10-weighted-vote: 확신도 가중 투표 (H16). `remote/eval_vllm.py --mode sc --n 8` 실행 — 스크립트가 개선되어 일반 다수결과 가중 투표를 한 번에 측정한다. 출력의 `[sc_n8]`와 `[sc_n8+weighted]` 두 수치를 기록하고 exp05(74.7%)와 비교. 베이스 모델(어댑터 없음) → 결과: 다수결 73.7%(356/483), 가중 투표 74.5%(360/483) — 같은 표본 내 +0.8%p 개선(H16 방향 확인, 기대 하단). exp05(74.7%)와 다수결 값 차이는 vLLM 비결정성 추정
- [x] exp09a-numina-prep: 외부 데이터 준비 (H12). `uv run python remote/prep_numina.py --take 30000` 실행 (datasets 라이브러리 필요 — 설치돼 있음). 채택 개수와 혼합 결과(외부+자체) 통계를 기록. GPU 불필요, 다운로드 수 GB → 결과: 외부 30000개 채택(data/numina.jsonl), 자체 12923개와 병합 총 42923개(data/sft_mix.jsonl). 이전 세션에 이미 실행돼 있던 것을 검증 후 기록 대행(재실행하지 않음)
- [ ] exp09b-qlora-mix: 외부 혼합 학습 (H12 — 자기증류 탈피 본명). `remote/train_qlora.py` CONFIG를 data_path=data/sft_mix.jsonl, output_dir=outputs/qlora_mix, lr=1e-4, epochs=1로 수정 후 학습 → `remote/eval_vllm.py --mode both --adapter outputs/qlora_mix/<최종디렉토리>` 평가. 베이스(69.4/74.7) 대비 판정. **이 실험이 실패하면 정체 3/3 — 근본 전환 문서 작성 단계로**
