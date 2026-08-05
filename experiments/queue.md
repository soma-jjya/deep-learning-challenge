# 실험 큐 (서버 러너가 위에서부터 순서대로 실행)

규칙: 로컬 Claude(계획자)가 실험을 추가하고, 서버 Claude(실행자)는 실행·기록·체크만 한다.
완료 표기: `- [x] ... → 결과: ...`

- [ ] exp04-rft-gen: `remote/generate_rft.py` 실행해 data/sft.jsonl 생성. 완료 후 "정답 도달 문제 수 / 채택 풀이 수"를 기록. sft.jsonl은 커밋하지 말고(용량) 통계만 기록
- [ ] exp05-eval-base: `remote/eval_vllm.py --mode both` 실행 (베이스 모델, 검증 500, greedy + SC n=8). exp02·03에 해당하는 AWS 측 기준점 확보
- [ ] exp06-qlora-r16: `remote/train_qlora.py` 기본 설정(r=16, lr=2e-4, ep=2)으로 SFT 학습 → 완료 후 `remote/eval_vllm.py --mode both --adapter outputs/qlora/<최종디렉토리>` 로 평가. wandb 링크와 loss 최종값도 기록
