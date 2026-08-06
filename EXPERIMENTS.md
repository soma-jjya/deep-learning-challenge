# 실험 기록 (Experiment Log)

모든 실험을 시간순으로 기록한다. 각 실험마다: 무엇을 바꿨는지 / 왜 / 결과 점수.
이 파일이 나중에 "실험 보고서"의 원본 자료가 된다. 근거 자료는 docs/research.md.

## 가설 백로그 (우선순위순 — 결과 나올 때마다 갱신)

| 가설 | 내용 | 기대 효과 | 근거 | 상태 |
|---|---|---|---|---|
| H1 | max 토큰 1024→2048 + 소수점 오추출 수정 | +2~4%p | exp01 오답 분석 | 노트북 02 반영 |
| H2 | Self-Consistency n=8 (temp 0.7 다수결) | +5~10%p | Wang et al. 2022, AIMO 우승 사례 | 노트북 02 반영 |
| H3 | RFT: 자체 생성 CoT 중 정답만 골라 QLoRA SFT | +5~15%p | RFT 논문, STaR | ❌ 2연속 실패 (exp06, exp06c) — 정체 2/3, lr·데이터량 조절로도 회복 안 됨 |
| H4 | 외부 CoT 데이터 혼합 (NuminaMath-CoT) | +3~8%p | NuminaMath 1위 솔루션 | 대기 |
| H5 | ~~TIR: 모델이 쓴 파이썬을 실행해 계산 보조~~ | - | - | ❌ **금지** (운영진 Q&A: 추론 시 코드 실행·툴 호출 불허) |
| H6 | GPTQ/AWQ 양자화 + vLLM으로 SC 샘플 수 증대 | 간접 (속도) | NuminaMath T4 최적화 | 속도 병목 시 |
| H7 | GRPO 강화학습 | 불확실 | DeepSeekMath | 후순위 |
| H8 | 멀티 LoRA 어댑터 앙상블 (같은 베이스, 다수결 결합) | +2~5%p | SC의 다양성 확장 | ⏳ 운영진 질의 중 |
| H9 | 검증자(verifier) 어댑터로 Best-of-N 선별 | +3~8%p | 소형모델+강한검증자 연구 | ⏳ 운영진 질의 중 |
| H10 | 반복 RFT: 학습된 모델로 데이터 재생성 → 재학습 (2~3라운드) | +3~8%p | STaR의 반복 루프 | ⏭️ 스킵 (exp08) — exp06c 어댑터가 베이스보다 낮아 조건 미충족 |
| H11 | SC 샘플 수 스케일링 (8→16→32) + temperature 탐색 | +1~4%p | 다수결은 표본이 클수록 안정 | ✅ 완료 (exp07) — n=8→16 +0.9%p로 체감, n=32는 비용 대비 보류 |
| H12 | 외부 CoT 데이터 혼합 비율 실험 (NuminaMath-CoT 정수답 부분집합 0/30/70%) | +3~8%p | NuminaMath 우승, H4 구체화 | 대기 (데이터 준비 스크립트 필요) |
| H13 | 오답 유형 분석 → 취약 유형 표적 데이터 증강 (상용 API 활용 허용 범위) | +2~6%p | 오답 분석 기반 커리큘럼 | exp05 오답 분석 후 |
| H14 | 풀이 길이 통제: 짧은 정답 풀이 우선 학습 (긴 풀이는 3B에 역효과) | +1~4%p | Small Models Struggle 논문 | ❌ 완료 (exp06c, lr완화와 결합) — 효과 없음, 여전히 베이스보다 낮음 |
| H15 | DPO: 같은 문제의 정답 풀이(chosen) vs 오답 풀이(rejected) 쌍 학습 | +2~6%p | RFT 부산물 데이터 재활용 | SFT 정체 시 |
| H16 | 로그확률 가중 투표 (모델 출력의 확신도로 표 가중치 — 모델 출력만 사용이라 규칙 적합) | +1~3%p | Entropy-weighted SC (AIMO-3) | ✅ 완료 (exp10) — 같은 SC n=8 표본 내에서 가중 투표가 일반 다수결보다 +0.8%p 높음(74.5% vs 73.7%), 기대 범위(+1~3%p) 하단에 근접 |

**정체 대비 근본 전환 후보** (3연속 정체 시 이 중에서 문서화 후 착수): GRPO 강화학습(H7) / 학습 데이터 전면 재구성(외부 데이터 중심) / 추론 파이프라인 재설계(다단계 자기수정) / 커리큘럼 학습(쉬운→어려운 순서 제어)

> 2026-08-03 공지 반영: 학습·검증에서 train 오류 627문항 제외, 리더보드는 filtered(831문항) 기준. 이전 리더보드 점수(exp01 0.648)와 새 기준 점수는 직접 비교 불가.
> 정형 기록은 `experiments/log.csv` (모든 실험·시행착오 필수 기재), 시각화는 `report.html`.

| # | 날짜 | 방법 | 미니 평가 (train 50) | 리더보드 | 노트북 |
|---|------|------|---------------------|----------|--------|
| 1 | 2026-07-31 | 베이스라인: Qwen2.5-3B-Instruct 그대로, greedy, max 1024 토큰, boxed 추출 | 66.0% (33/50) | **0.648** | kaggle/01_baseline_inference.ipynb |
| 4 | 2026-08-05 | RFT 데이터 생성 (AWS, `remote/generate_rft.py`) — 대상 16,500문제(검증 500·오류 문항 사후필터 전), 문제당 6개 샘플(temp 0.8), 정답 도달 문제 13,190개(79.9%), 채택 풀이 36,733개 → data/sft.jsonl | - | - | remote/generate_rft.py |
| 4b | 2026-08-05 | exp04 sft.jsonl 사후 필터링 — 오류 문항 627개 기준 라인 제거: 36,733줄 → 36,144줄 (589줄 제거) | - | - | - |
| 5 | 2026-08-05 | 베이스 모델 AWS 평가 (`remote/eval_vllm.py --mode both`) — 검증 483문항(500 중 오류 문항 제외), greedy 69.4%, SC n=8 74.7% | - | - | remote/eval_vllm.py |
| 6 | 2026-08-05 | **QLoRA SFT 1차 (H3) — 실패** — r16/lr2e-4/ep2, RFT 36,144 풀이 학습. greedy 68.3%(-1.1%p), SC n8 73.5%(-1.2%p). [wandb](https://wandb.ai/loonaticvibe2-11-jin-jason/huggingface/runs/o5zlvcyp) | - | - | remote/train_qlora.py |
| 6b | 2026-08-06 | 리더보드 제출 파일 생성 — exp06 어댑터가 베이스보다 나빠 **어댑터 없이** `remote/make_submission.py --n 8 --tag base` 실행 (SC n=8, 831문항) → results/submission_base.csv | 74.7% | **0.77015** ✅ | remote/make_submission.py |
| 7 | 2026-08-06 | SC 스케일 (H11, 베이스) — n=4: 72.7%, n=8: 74.7%, **n=16: 75.6%** (+0.9%p vs n8). 표본 2배당 ~+1%p의 완만한 수익 곡선 | 75.6% (n16) | - | remote/eval_vllm.py |
| 7 | 2026-08-06 | SC 샘플 수 스케일링 (H11) — 베이스 모델, `remote/eval_vllm.py --mode sc --n 16`/`--n 4` (검증 483문항) | n=4 72.7%(351) / n=8 74.7%(361,exp05) / n=16 75.6%(365) | - | remote/eval_vllm.py |
| 6c | 2026-08-06 | **QLoRA SFT 재시도 (H3+H14) — 재실패** — r16/lr5e-5/ep1(완만), 문제당 최단 풀이 1개. greedy 67.9%(-1.5%p), SC n8 73.3%(-1.4%p) | 67.9%(greedy)/73.3%(SC) | - | remote/train_qlora.py |
| 8 | 2026-08-06 | 반복 RFT (H10) — exp06c 어댑터가 베이스보다 낮아 조건 불충족, **스킵** | - | - | - |
| 10 | 2026-08-06 | 확신도 가중 투표 (H16, 베이스) — `remote/eval_vllm.py --mode sc --n 8`, 같은 표본에서 일반 다수결 73.7%(356/483) vs 가중 투표 **74.5%**(360/483) | 74.5%(가중) | - | remote/eval_vllm.py |
| 9a | 2026-08-06 | NuminaMath-CoT 외부 데이터 준비 (H12) — `remote/prep_numina.py --take 30000`, 정수답·boxed·길이(50~2500자) 필터 + 검증셋 문제 텍스트 제외 → 외부 30,000개 채택, 자체(`data/sft_short.jsonl`) 12,923개와 병합·셔플 → `data/sft_mix.jsonl` 42,923개 | - | - | remote/prep_numina.py |

## 실험 9a: NuminaMath 외부 데이터 준비 (2026-08-06, AWS)

- **설정**: `remote/prep_numina.py --take 30000` — AI-MO/NuminaMath-CoT(Apache 2.0, 원본 859,494개)에서 셔플 후 순회하며 채택: 풀이 길이 50~2500자, `boxed` 포함, `extract_answer`로 정수 답 추출 가능, 검증 세트(seed=123, 500문항) 질문 텍스트와 겹치지 않음. 채택된 외부 데이터를 자체 RFT 최단 풀이 데이터(`data/sft_short.jsonl`, 문제당 1개)와 병합·셔플
- **결과**: 외부 데이터 **30,000개** 채택 → `data/numina.jsonl`. 자체 12,923개와 합쳐 총 **42,923개** → `data/sft_mix.jsonl` (H12 학습용 입력, exp09b에서 사용)
- **결과 파일**: `data/numina.jsonl`(30,000줄), `data/sft_mix.jsonl`(42,923줄), 로그 `exp09a.log`
- **사고 기록**: 이 실행도 이전 서버 세션에서 이미 완료돼 있었으나(파일·로그 08-06 08:07 생성) 기록·커밋 전에 세션이 중단된 것으로 추정. 이번 세션에서 산출물을 검증(파일 라인 수 일치 확인) 후 기록만 대행, 재실행하지 않음. 같은 세션에서 이어서 `remote/train_qlora.py`로 exp09b 학습(`outputs/qlora_mix/qlora_r16_lr0.0001_ep1_final`, train loss 0.3994, [wandb fvh36wox](https://wandb.ai/loonaticvibe2-11-jin-jason/huggingface/runs/fvh36wox))까지도 이미 완료돼 있었음 — 평가만 이번 세션에서 실행
- **다음**: exp09b 평가(`remote/eval_vllm.py --mode both --adapter outputs/qlora_mix/qlora_r16_lr0.0001_ep1_final`)

## 실험 10: 확신도 가중 투표 (2026-08-06, AWS)

- **설정**: `remote/eval_vllm.py --mode sc --n 8` (베이스 모델, 어댑터 없음), 검증 483문항(exp05·07과 동일 세트, seed=123). SC 샘플(temperature=0.7, top_p=0.8, seed=42)에 대해 일반 다수결과 함께, 같은 생성 결과에서 확신도(`cumulative_logprob / 토큰수`)로 softmax 가중(scale=2.0) 후 투표하는 가중 다수결(H16)을 동시에 측정
- **결과**: 일반 다수결 **73.7% (356/483)**, 확신도 가중 투표 **74.5% (360/483)** — 같은 표본 내 비교로 가중 투표가 **+0.8%p** 높음
- **의미**: H16(기대 +1~3%p) 방향은 맞지만 기대 하단에 근접하는 수준의 개선. 다만 이번 실행의 일반 다수결 값(73.7%)은 exp05에서 측정한 SC n=8(74.7%, 361/483)보다 낮게 나왔는데, 두 실행 모두 동일한 시드(temp=0.7/top_p=0.8/seed=42)·동일 모델·동일 검증셋을 사용했음에도 차이가 남 → vLLM 연속 배칭(continuous batching)의 실행 간 비결정성으로 추정(±1%p 내외 노이즈). 가중 투표 자체의 효과는 "같은 실행 내" 비교(73.7%→74.5%)로 판단하는 것이 안전
- **결과 파일**: `results/eval_base.json` (`sc_n8`, `sc_n8_weighted` 필드), 로그 `eval10.log`
- **사고 기록**: 이번 실행은 이전 서버 세션에서 이미 완료되어 있었으나(로그·결과 파일 존재, 08-06 08:02 커밋) EXPERIMENTS.md·log.csv·queue.md 기록과 푸시가 되지 않은 채 남아 있었음(추정: 서버 Claude 사용량 한도로 세션 중단). 결과를 로그로 검증 후 이번 세션에서 기록만 대행, 재실행하지 않음
- **다음**: exp09a(NuminaMath 외부 데이터 준비, H12) → exp09b(외부 혼합 QLoRA)

## 실험 6c: QLoRA SFT 재시도 — 완만한 학습 + 최단 풀이 (2026-08-06, AWS)

- **설정**: exp06 실패 대응 1단계. `data/sft.jsonl`에서 문제당 가장 짧은 정답 풀이 1개만 남긴 `data/sft_short.jsonl` 생성 → `remote/train_qlora.py`를 lr=5e-5(exp06의 1/4), epochs=1(exp06의 절반)로 학습(train loss 0.2429, [wandb sv8lkhho](https://wandb.ai/loonaticvibe2-11-jin-jason/huggingface/runs/sv8lkhho)) → 어댑터 `outputs/qlora_gentle/qlora_r16_lr5e-05_ep1_final` → `remote/eval_vllm.py --mode both --adapter ...` (검증 483문항, exp05·06·07과 동일 세트)
- **결과**: greedy **67.9% (328/483)**, SC n=8 **73.3% (354/483)** — 베이스(69.4%/74.7%) 대비 greedy −1.5%p, SC −1.4%p. exp06(68.3%/73.5%)과 비교해도 **거의 동일하거나 소폭 더 낮음**
- **의미**: 학습률을 1/4로, epoch을 절반으로, 학습 데이터도 문제당 최단 풀이 1개로 줄였음에도 실패 폭이 줄지 않음 → **원인 가설 1(학습률 과함)은 주 원인이 아니었을 가능성이 높음**. 남은 원인 가설(2. 자기증류로 출력 다양성 감소, 3. 이미 푸는 문제만 학습해 실력 확장 없음) 쪽에 무게가 실림
- **정체 카운트 갱신**: QLoRA SFT 계열 시도 2연속 실패 (exp06, exp06c) → **정체 카운트 2/3**. 다음 QLoRA류 시도가 또 실패하면 근본 전환(GRPO/외부데이터 중심 재구성/DPO 등) 착수
- **후속 결정 (exp08)**: 반복 RFT(H10)는 "exp06c 어댑터가 베이스보다 좋을 때만 실행" 조건이었으나 조건 미충족 → **스킵**. 대신 백로그의 H12(외부 데이터 혼합)·H15(DPO)가 다음 유력 후보
- **결과 파일**: `results/eval_outputs_qlora_gentle_qlora_r16_lr5e-05_ep1_final.json`, 어댑터는 삭제하지 않고 보존
- **다음**: 큐 비어 있음 — 로컬 Claude가 H12/H15 중 다음 실험을 큐에 등록해야 함

## 실험 6: QLoRA SFT 1차 — 실패 분석 (2026-08-05→06)

- **결과**: 베이스 대비 greedy −1.1%p, SC −1.2%p. 학습 자체는 정상 종료(loss 0.166, 2 epoch)했으나 **낮아진 loss가 오히려 과적합 신호**
- **원인 가설 3가지**:
  1. **학습률 과함**: 2e-4 × 2 epoch은 3B 모델에 공격적 — 자기 풀이를 암기하는 수준까지 loss가 내려감
  2. **자기증류의 한계**: 자기가 만든 정답 풀이만 다시 배우면 새 정보가 없고, 출력 다양성이 줄어 **SC 다수결의 재료(다양한 풀이 경로)가 오히려 감소**
  3. **데이터 편향**: 이미 푸는 80% 문제의 풀이만 학습 — 못 푸는 20%는 데이터에 없어서 실력 확장이 안 됨
- **대응 (연결 체인)**: exp06c = 완만한 학습(lr 5e-5, 1 epoch) + 문제당 최단 풀이 1개(H14, 3B에 유리) → 그래도 안 되면 가설 2·3을 정면 공략: 외부 데이터 혼합(H12)·DPO(H15)로 전환. 정체 카운트 1/3
- **사고 기록**: 러너가 git pull 충돌로 사망해 평가 후 기록이 누락됐었음 → 러너에 자동 커밋 후 재시도 로직 추가

## 실험 6b: 리더보드 제출 파일 생성 (2026-08-06, AWS)

- **설정**: `remote/make_submission.py --n 8 --tag base` — exp06 QLoRA 어댑터가 베이스보다 하락했으므로 **어댑터 없이 베이스 모델**로 생성. SC n=8(temperature=0.7, top_p=0.8), 리더보드 831문항 전체(`deep_chal_math_leaderboard_filtered.csv`) 대상
- **결과**: `results/submission_base.csv` 831행 생성 완료, 정수 답 831개 전부 채움(결측 없음 확인). 실제 리더보드 점수는 사용자가 Kaggle에 제출해야 확인 가능
- **다음**: exp07(SC 스케일 실험) → exp06c(완만한 QLoRA 재시도)

## 실험 7: SC 샘플 수 스케일링 (2026-08-06, AWS)

- **설정**: `remote/eval_vllm.py --mode sc --n 16`, 이어서 `--n 4` — 베이스 모델(어댑터 없음), 검증 483문항(exp05와 동일 세트), temperature=0.7, top_p=0.8
- **결과**: n=4 **72.7%** (351/483) / n=8 **74.7%** (361/483, exp05) / n=16 **75.6%** (365/483)
- **의미**: n=4→8 구간(+2.0%p)이 n=8→16 구간(+0.9%p)보다 이득이 커 **수확 체감** 확인(H11 가설과 일치, 다만 32까지는 아직 미탐). 추론 비용(샘플 수 비례)을 고려하면 제출용으로는 n=8이 비용 대비 합리적 선택 — n=16은 소폭 이득(+0.9%p) 대비 2배 비용
- **결과 파일**: `results/eval_base_sc_n4.json`, `results/eval_base_sc_n16.json`
- **사고 기록**: 실행 자체는 이전 서버 러너 세션에서 완료됐으나(로그 `logs_exp07_n16.log`, `logs_exp07_n4.log` 확인) 기록 단계 전에 세션이 종료되어 이번 세션에서 결과를 확인 후 기록만 대행
- **다음**: exp06c(완만한 QLoRA 재시도, H14 결합)

## 실험 4: RFT 데이터 생성 (2026-08-05, AWS)

- **설정**: `remote/generate_rft.py`, 베이스 모델 Qwen2.5-3B-Instruct, vLLM 샘플링(temperature=0.8, top_p=0.95, max_tokens=2048), 문제당 n_samples=6, 문제당 정답 풀이 최대 3개 채택(max_keep), chunk_size=500(중단-재개 지원)
- **대상**: train 17,000문제 중 검증 500문제(seed=123) 제외한 16,500문제. 이 시점엔 아직 오류 문항(627개) 사후 필터 전 — exp04b에서 제거
- **결과**: 16,500/16,500 처리 완료. **정답 도달 문제 13,190개 (79.9%)**, 채택 풀이(assistant CoT) 총 **36,733개** → `data/sft.jsonl`에 저장 (커밋 대상 아님, 통계만 기록)
- **exp04b 필터링**: `deep-learning-challenge-2026/train_filtered_ids.csv`(오류 문항 627개)의 id에 해당하는 라인을 id 필드 기준으로 제거. **36,733줄 → 36,144줄 (589줄 제거)**. 이후 `data/sft.jsonl`은 필터된 버전으로 교체됨
- **다음**: exp05(베이스 모델 AWS 평가) → exp06(QLoRA r16 SFT, 필터된 sft.jsonl 사용)

## 실험 5: 베이스 모델 AWS 평가 (2026-08-05, AWS)

- **설정**: `remote/eval_vllm.py --mode both` (베이스 모델, 어댑터 없음), 검증 500문제(seed=123)에서 오류 문항(train_filtered_ids.csv) 자동 제외 → **유효 검증 문항 483개**
- **결과**: greedy(temperature=0, max_tokens=2048) **69.4% (335/483)**, SC n=8(temperature=0.7, top_p=0.8) **74.7% (361/483)**
- **의미**: exp02·03(Kaggle, 구 500문항 기준)과는 검증 세트 구성이 달라 직접 비교 불가하지만, AWS 환경·필터된 483문항 기준의 새 베이스라인 확보. exp06(QLoRA) 이후 이 수치와 비교해 SFT 효과 측정
- **결과 파일**: `results/eval_base.json`
- **다음**: exp06-qlora-r16

## 실험 1: 베이스라인 (2026-07-31)

- **설정**: 파인튜닝 없음. fp16, T4 x2, greedy 디코딩(do_sample=False), max_new_tokens=1024, batch 16
- **프롬프트**: system에 "step by step + 최종 정수 답을 \boxed{}에" 지시
- **후처리**: \boxed{} 마지막 값 → 없으면 마지막 정수 → 없으면 0
- **결과**: train 50문제 미니 평가 66.0%
- **발견한 문제점** (다음 실험에서 개선할 것):
  1. 긴 풀이가 1024 토큰에서 잘려 \boxed{}를 못 쓰는 경우 발생 → 토큰 한도 2048로 증가 필요
  2. 폴백 추출이 소수의 일부를 정수로 착각 (예: "38.585"에서 585를 뽑음) → 소수점 뒤 숫자 제외 로직 필요
  3. **(사후 발견)** GitHub→Kaggle 복사 과정에서 백슬래시가 한 겹 벗겨져, 실행된 사본에서는 \boxed 추출 정규식과 프롬프트의 \boxed{} 지시가 깨져 있었음. 즉 0.648은 "마지막 정수 폴백"만으로 낸 점수 → 노트북 02부터 모든 패턴을 백슬래시 없이(chr(92) 조립) 작성해 복사 경로와 무관하게 동작. exp02 상승분에는 이 수리 효과가 포함될 것
