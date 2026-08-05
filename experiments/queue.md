# 실험 큐 (서버 러너가 위에서부터 순서대로 실행)

규칙: 로컬 Claude(계획자)가 실험을 추가하고, 서버 Claude(실행자)는 실행·기록·체크만 한다.
완료 표기: `- [x] ... → 결과: ...`

- [ ] exp04-rft-gen: `remote/generate_rft.py` 실행해 data/sft.jsonl 생성. 완료 후 "정답 도달 문제 수 / 채택 풀이 수"를 기록. sft.jsonl은 커밋하지 말고(용량) 통계만 기록
- [ ] exp04b-filter-sft: exp04가 오류 문항 제외 전에 시작됐으므로 사후 필터링. `deep-learning-challenge-2026/train_filtered_ids.csv`의 id에 해당하는 라인을 data/sft.jsonl에서 제거(각 라인의 "id" 필드 기준). 제거 전/후 라인 수를 기록
- [ ] exp05-eval-base: `remote/eval_vllm.py --mode both` 실행 (베이스 모델, greedy + SC n=8). 스크립트가 검증 500에서 오류 문항을 자동 제외함 — 출력되는 유효 검증 문항 수를 함께 기록. exp02·03에 해당하는 AWS 측 기준점 확보
- [ ] exp06-qlora-r16: `remote/train_qlora.py` 기본 설정(r=16, lr=2e-4, ep=2)으로 SFT 학습 (데이터는 exp04b의 필터된 sft.jsonl) → 완료 후 `remote/eval_vllm.py --mode both --adapter outputs/qlora/<최종디렉토리>` 로 평가. wandb 링크와 loss 최종값도 기록
- [ ] exp07-sc-scale: exp06 어댑터로 SC 스케일 실험 (H11). `remote/eval_vllm.py --mode sc --adapter <exp06 어댑터> --n 16` 실행, 이어서 `--n 4`도 실행. exp06의 n=8 결과와 함께 n별 정확도 표로 기록 (표본 수 대비 이득 곡선 확인 목적)
- [ ] exp08-rft-round2: 반복 RFT (H10). `remote/generate_rft.py`를 exp06 어댑터로 실행하되 출력은 data/sft_round2.jsonl로 — 스크립트가 어댑터 로드를 지원하지 않으면 CONFIG를 수정해 vLLM LoRA 로드를 추가할 것 (enable_lora, LoRARequest — eval_vllm.py 참고). 완료 후 정답 도달률을 exp04와 비교 기록 (학습된 모델이 더 많은 문제의 풀이를 찾는지 확인)
