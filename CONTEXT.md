# CONTEXT.md — 프로젝트 맥락 관리 (항상 이 파일부터 읽기)

> 세션이 바뀌어도 이 파일 하나로 전체 맥락을 복원한다. 상태가 바뀔 때마다 즉시 갱신할 것.

## 목표

- **로컬 검증 정확도 85%** 달성 (현재 리더보드 0.648)
- 방법 불문: 후처리, 프롬프트, Self-Consistency, QLoRA SFT, GRPO 등 전부 후보
- 모든 실험은 EXPERIMENTS.md에, 시각화·보고서는 report.html에 기록

## 작업 루프 (역할 분담)

1. Claude: 가설 수립 → 노트북 제작 → GitHub 푸시
2. 사용자: Kaggle에서 노트북 실행 (GPU T4 x2, Internet On) → 출력 숫자/에러를 채팅에 붙여넣기
3. Claude: 결과 분석 → EXPERIMENTS.md·report.html 갱신 → 다음 가설 반영한 노트북 수정
4. 좋은 결과만 리더보드 제출 (description: `expNN | 방법 | local XX%`)

## 확정 전략 (2026-08-04, 사용자 승인)

**하이브리드**: 추론·평가·제출 = Kaggle (무료, 최종 추론 재현성도 Kaggle 환경 기준) / RFT 데이터 생성·QLoRA 학습 = AWS (9시간 벽·T4 속도 병목 회피, 스팟 총 $30~80 예상). 어댑터는 AWS→Kaggle로 가져와 추론.

## 현재 상태 (2026-08-04)

- exp01 베이스라인 완료: 로컬 66% (50문제), **리더보드 0.648** — 로컬≈LB 확인됨
- 노트북 02 백슬래시 버그 수정 완료 (chr(92) 조립 방식) → exp02·03 재실행 대기
- RFT 파이프라인 준비 완료: `remote/answer_extract.py`(노트북 02와 동일 추출기 공유), `remote/generate_rft.py`(vLLM로 문제당 6개 풀이 샘플링 → 정답만 채택 → data/sft.jsonl, 중단-재개 지원) → AWS 인스턴스 뜨면 바로 실행 가능
- **AWS 실험 인프라 구축 중** (remote-finetune-session.md 기반): `aws/` 폴더에 launch.ps1(인스턴스 시작)·bootstrap.sh(자동 세팅)·README.md(절차), `remote/train_qlora.py`(Unsloth QLoRA 스크립트) 준비됨. 로컬에 AWS CLI 설치 완료
- **AWS 자율 실험 인프라 완성 (2026-08-05)**: 격리 3중 장치(태그 조건 IAM 정책 `aws/iam-policy.json` + 전용 VPC + state.json), 자율 러너(`remote/run_experiments.sh` — experiments/queue.md를 서버 Claude가 순서대로 실행·기록·푸시), 유휴 30분 자동 stop(`remote/watchdog_autostop.sh`), ntfy 폰 알림. 실험 큐에 exp04(RFT 생성)·exp05(베이스 평가)·exp06(QLoRA r16) 등록됨. 대기: 사용자의 IAM 사용자 생성 + `aws configure --profile ajudl` (aws/README.md 0장)
- **역할 분담 확정**: 로컬 Claude = 계획자(큐에 실험 정의, 결과 분석, 보고서), 서버 Claude = 실행자(큐 실행·기록만, 임의 실험 추가 금지)
- **Kaggle API 자동 실행 파이프라인 구축 중**: kaggle CLI 설치됨, `kaggle/kernel-metadata.json`(템플릿 — KAGGLE_USERNAME·COMPETITION_SLUG 치환 필요) + `scripts/kaggle_run.ps1`(푸시→폴링→로그 회수). 대기: 사용자의 kaggle.json(API 토큰, `C:\Users\82108\.kaggle\`에 저장)과 대회 URL(slug)
- **AWS는 보류, 당분간 Kaggle 사용** (2026-08-03 결정). AWS로 넘어갈 때 필요한 것: ① 계정·액세스 키 → `aws configure` ② GPU 쿼터(G/VT vCPU ≥4) 증가 신청 ③ `claude setup-token` ④ GitHub fine-grained PAT (레포 private — bootstrap이 GITHUB_TOKEN으로 클론)
- 서버 확정 스펙: g5.xlarge(A10G 24GB) 서울 리전 권장, 안 쓸 때 stop 필수

## 핵심 결정·제약 (변경 시 여기 갱신)

- 실행 환경: Kaggle T4 x2 (로컬 PC는 6GB라 불가). T4는 bf16 미지원 → fp16
- 검증 세트: train에서 500문제 고정 (`random_state=123`) — **이후 SFT 학습 데이터에서 반드시 제외**
- 미니 평가 50문제는 폐기 (오차 ±13%p) → 500문제로 통일
- 제출 컬럼: 소문자 `id` (문서의 `ID`는 오기)
- MoE 제외 (베이스 모델 고정 규칙 위반 소지)
- SFT 함정: train엔 최종 답만 있음 → 풀이(CoT) 없이 "문제→답" 학습 금지. Rejection Sampling으로 CoT 생성 필요
- 외부 데이터 사용 시: 무료 공개만 가능, README '사용 데이터'에 출처 즉시 기재

## 파일 지도

| 파일 | 역할 |
|---|---|
| CONTEXT.md | 이 파일. 맥락 복원용 |
| prd.md | 대회 규칙 요약 (제출물 필수/권장 포함) |
| EXPERIMENTS.md | 실험 기록 + 가설 백로그 |
| report.html | HTML 실험 보고서 (그래프·급상승 이유·참고 문헌) — 브라우저로 열기 |
| docs/research.md | 리서치 결과·출처 |
| kaggle/01_baseline_inference.ipynb | exp01 베이스라인 |
| kaggle/02_eval_and_selfconsistency.ipynb | exp02·03 (개선 후처리 + SC) |
| requirements.txt | 실행 환경 (Kaggle에서 freeze한 파일은 kaggle_freeze/) |

## 검증 제출물 체크리스트 (수상 후보 대비)

| 항목 | 구분 | 상태 |
|---|---|---|
| 최종 모델 체크포인트 | 필수 | ⬜ SFT 단계에서 Kaggle Dataset으로 저장 예정 |
| 학습 코드 (전체 파이프라인) | 필수 | ⬜ SFT 노트북에서 |
| 추론 코드 (생성→후처리→CSV) | 필수 | ✅ kaggle/01, 02 노트북 |
| 사용 데이터 목록 | 필수 | ✅ README에 유지 중 |
| 실행 환경 (requirements.txt) | 필수 | ✅ + 노트북에 pip freeze 셀 |
| 학습 로그 (wandb/tensorboard) | 권장 | ⬜ SFT 시작 시 wandb 연동 |
| 실험 보고서 | 권장 | ✅ report.html + EXPERIMENTS.md |
| README (재현 가이드) | 권장 | ✅ 유지 중 |
