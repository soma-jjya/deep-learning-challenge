# 아주 소중한 딥러닝 챌린지 2026 — PRD

## 목표

범용 모델 **Qwen/Qwen2.5-3B-Instruct**를 출발점으로, 처음 보는 수학 문제의 **정수 답**을 정확히 추론하는 모델을 만든다.

- 평가 지표: **Accuracy (Exact Match)** — 제출한 정수와 정답이 정확히 일치해야 정답
- 모든 문제의 정답은 정수

## 핵심 규칙

### 모델
- 베이스 모델은 `Qwen/Qwen2.5-3B-Instruct` 고정 (HuggingFace)
- 다른 모델(Qwen2.5-Math, DeepSeek-R1, Llama 등)을 베이스로 쓰거나 가중치를 병합(merge)하는 것 금지
- 추론 시 외부 모델 호출·앙상블 금지
- Pre-training은 금지, Fine-tuning만 허용

### 허용되는 학습 기법
- Full Fine-Tuning, LoRA, QLoRA 등 PEFT
- SFT (Supervised Fine-Tuning)
- RL 기반 학습 (GRPO, DPO, PPO, KTO 등)
- 데이터 증강, 커리큘럼 학습
- 양자화 (Quantization)

### 데이터
- 주최 측 제공 학습 데이터가 기본
- 외부 공개 데이터셋 사용 자유 (모든 참가자가 무료로 접근 가능한 것만; 유료·비공개 데이터 금지)
- 사용한 외부 데이터셋은 최종 제출 시 목록 명시 필요
- test 데이터를 학습에 사용하는 것 금지
- 학습 데이터 구축 목적의 상용 API 사용은 허용 (예: GPT-4로 풀이 생성, 데이터 증강)
- 상용 API로 테스트 문제의 답을 직접 생성하는 것 금지
- 리더보드 점수로 정답을 역추적(probing)하는 행위 금지

### 추론
- 추론 시 인터넷 접속 차단 (외부 API 호출·웹 검색 불가) — 모든 추론은 로컬 수행
- 테스트 타임 기법(Majority Voting, Self-Consistency, Best-of-N 등) 자유롭게 사용 가능

## 데이터셋

| 파일 | 용도 |
|---|---|
| `deep_chal_math_dataset_train.csv` | 학습용 train set |
| `deep_chal_math_dataset_leaderboard.csv` | 실시간 리더보드 평가용 (answer 비어 있음) |
| `deep_chal_math_dataset_test.csv` | 최종 평가용 — 8/31 00:00 공개, 8/31 00:00~23:59 내 제출 |

### 필드
| 필드 | 설명 |
|---|---|
| `id` | 고유 식별자. 제출 파일과 정답 매칭 기준 — 변경 금지 |
| `question` | 수학 문제 텍스트 (자연어 + LaTeX 수식 혼합 가능) |
| `answer` | 최종 정답 (정수). Test에서는 비어 있음 |

예시:
```json
{
  "id": "train-000000",
  "question": "What is the molecular weight of some moles of Aluminum chloride if the molecular weight of 3 moles is 396?",
  "answer": "132"
}
```

## 평가

- **실시간 리더보드 (Public)**: Test의 30%로 채점. 참고용 — 최종 순위에 영향 없음
- **최종 리더보드 (Private)**: Test의 나머지 70%로 채점. 최종 순위 결정
- 모델 출력에는 풀이 과정·수식·설명이 섞여 있을 수 있으므로, **최종 답 추출 후처리(post-processing)는 직접 구현**해야 함

## 제출 형식

- 파일명: `submission.csv`
- 컬럼: `id`, `answer` (answer는 반드시 정수만) — 문서에는 `ID`로 표기돼 있으나 실제 채점기는 소문자 `id` 요구 (2026-08-01 제출로 확인)
- 모든 문제에 답 필수 — 빈 값은 오답 처리

```csv
ID,answer
prob_0001,42
prob_0002,7
prob_0003,125
```

## 수상 후보 시 제출물 (재현 검증 대비)

필수:
- 최종 모델 체크포인트 (베이스 모델·3B 아키텍처 확인)
- 학습 코드 — 전체 파이프라인 (모델 출발점, 데이터 처리, 학습 기법)
- 추론 코드 — 생성 → 후처리 → CSV (오프라인 추론, 외부 API 미사용 확인)
- 사용 데이터 목록 (추가 외부 데이터 출처 명시)
- 실행 환경 (requirements.txt 등 재현 환경 구성)

권장:
- 학습 로그 (wandb / tensorboard — 과정 투명성)
- 실험 보고서 (접근 방법·전략 설명)
- README (실행 순서, 랜덤 시드 등 재현 가이드)

주최 측이 직접 코드를 실행해 submission.csv 재현 여부를 검증. 재현 불가 시 수상 취소 가능 → **처음부터 재현 가능하게 코드/환경/시드/로그를 관리할 것**

## 일정

| 일정 | 날짜 |
|---|---|
| 문제 공개 & 챌린지 시작 | 2026.07.31 |
| 챌린지 종료 | 2026.08.30 |
| 최종 test 공개 및 제출 | 2026.08.31 00:00 ~ 23:59 |
| 평가 및 검증 | ~2026.09.20 |
| 수상자 발표 | 2026.09.28 |

- 최종 12팀은 발표 진행: 모델 성능 50% + 발표 평가 50%로 최종 9팀 수상
- 디스코드: https://discord.gg/JhWDr73g65
