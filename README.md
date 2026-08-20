# 아주 소중한 딥러닝 챌린지 2026 — 수학 추론

Qwen2.5-3B-Instruct를 베이스로 수학 문제의 정수 답을 추론하는 대회 프로젝트.
규칙·평가 방식은 [prd.md](prd.md), 실험 이력은 [EXPERIMENTS.md](EXPERIMENTS.md) 참조.

## 저장소 구조

```
deep-learning-challenge-2026/   # 대회 제공 데이터 (train 17k / leaderboard 1k / test 2k)
kaggle/                         # Kaggle 노트북 (실행 환경: GPU T4 x2)
  01_baseline_inference.ipynb   # exp01 베이스라인: 파인튜닝 없이 추론 → submission.csv
  02_eval_and_selfconsistency.ipynb  # exp02·03: 고정 검증 500 + 개선 후처리 + Self-Consistency
CONTEXT.md                      # 프로젝트 맥락 관리 (진행 상태·결정 사항 — 여기부터 읽기)
prd.md                          # 대회 규칙 요약
EXPERIMENTS.md                  # 실험 기록 + 가설 백로그
report.html                     # 실험 보고서 (그래프·참고 문헌) — 브라우저로 열기
docs/research.md                # 리서치 노트 (논문·사례 출처)
requirements.txt                # 실행 환경
```

## 재현 방법

1. Kaggle 대회 페이지에서 New Notebook 생성, 해당 `.ipynb` 업로드
2. Settings: **Accelerator = GPU T4 x2**, **Internet = On**
3. Run All → `/kaggle/working/submission.csv` 생성

- 랜덤 시드: 미니 평가 샘플링은 `random_state=42`. 추론은 greedy(do_sample=False)라 시드 무관하게 결정적
- 라이브러리: Kaggle 기본 환경 (transformers, torch) 그대로 사용. 파인튜닝 단계부터 requirements.txt 관리 예정

## 사용 데이터

- 대회 제공 데이터 (train 17,000 / leaderboard_filtered 831 — 운영진 공지의 오류 문항 627개는 학습·검증에서 제외)
- [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) (Apache 2.0, 무료 공개) — 정수 답 부분집합을 SFT 혼합 학습에 사용 (`remote/prep_numina.py`로 추출, 검증 세트와 문항 중복 제거)
- 자체 생성 데이터 ①: 베이스 모델의 Rejection Sampling 풀이 약 36,000개 (`remote/generate_rft.py`) — **원천 문제는 대회 train 한정**
- 자체 생성 데이터 ②: 강한 교사 모델(Claude, 상용 API)이 작성한 모범 풀이 683개 (`api/gen_teacher_*.py`) —
  **학습 데이터 생성 목적으로만 사용(규칙 5.3.a 근거)이며, 최종 제출 모델에는 미사용** (교사 증류 실험 exp53은 성능 하락으로 기각)

### 데이터 출처 준수 확인 (2026-08-19 감사)

**증류·RFT의 원천 문제는 전량 대회 train 한정이다. test/leaderboard 문제는 어떤 외부 API·서비스에도
입력된 적이 없다** (규칙 5.1.b 및 5.3.b/c 준수). 검증 근거:
- 교사 대상 목록 전 행이 train ID: `data/hard_problems.csv` 2,967행 / `data/hard_problems_top.csv` 1,200행 /
  `data/desktop{,_top}/gold.csv` — **모두 `train-*` 접두** (leaderboard는 `val-*` 접두라 혼입 시 즉시 식별됨)
- RFT 입력: `remote/generate_rft.py`가 `deep_chal_math_train.csv`만 읽음
- `api/` 디렉토리 전체에 leaderboard/test 참조 0건 (전수 grep)
- 로컬 검증셋 483문항은 train에서 seed 고정 분할 (gold 라벨 보유가 그 증거)
- 최종 제출 모델 = **베이스 원본(Qwen2.5-3B-Instruct), 어댑터·추가 학습 미적용** — 학습된 가중치
  제출물이 없는 것이 정상임 (실험용 어댑터는 별도 보관, 추론 시 외부 API·도구·인터넷 미사용)
