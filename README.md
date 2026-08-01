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

- 대회 제공 데이터만 사용 (외부 데이터 추가 시 여기에 출처 명시)
