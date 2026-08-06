# CONTEXT.md — 프로젝트 맥락 관리 (항상 이 파일부터 읽기)

> 세션이 바뀌어도 이 파일 하나로 전체 맥락을 복원한다. 상태가 바뀔 때마다 즉시 갱신할 것.

## 목표

- **로컬 검증 정확도 85%** 달성 (현재 리더보드 0.648)
- 방법 불문: 후처리, 프롬프트, Self-Consistency, QLoRA SFT, GRPO 등 전부 후보
- 모든 실험은 EXPERIMENTS.md에, 시각화·보고서는 report.html에 기록

## 실험 운영 정책 (2026-08-05 사용자 지시 — 장기 유지)

1. **정체 시 근본 전환**: 유의미한 성능 변화 없는 실험이 **3연속**이면, 점진 개선을 멈추고 근본적으로 다른 학습·튜닝 방안을 문서로 작성 후 테스트한다 (예: 데이터 구성 전면 교체, 학습 패러다임 변경 SFT→DPO/GRPO, 추론 전략 재설계)
2. **가설 풀 상시 3개 이상**: EXPERIMENTS.md 백로그에 검증 가능한 가설을 항상 3개 이상 유지. 하나가 해소되거나 증가폭이 줄면 다음 가설로 넘어가고 새 가설을 보충. 하나의 흐름으로 이어지는 가설 체인(예: RFT→QLoRA→반복 RFT)은 연속 실행
3. **기록 체계**: 모든 실험은 ① `experiments/log.csv`(정형 데이터, 시행착오 포함) ② EXPERIMENTS.md(서술) ③ report.html(시각화 — 결과 분석 시점마다 갱신)의 3중 기록. 실패·중단도 기록한다
4. **긴 작업 처리**: GPU 12시간↑ 또는 비용 큰 작업은 사용자에게 선택지로 질문하되, **답변을 기다리는 동안 다른 큐 실험을 계속 돌린다** — 서버가 노는 상태 금지
5. **탐색 범위**: 데이터셋 단(오류 필터, 외부 데이터, 증강, 커리큘럼)부터 학습론(LoRA 설정, DPO/GRPO, 반복 RFT), 추론 전략(SC 스케일, 프롬프트, 투표 방식)까지 넓게. 안전한 것만 반복하지 말 것

## 작업 루프 (역할 분담)

1. Claude: 가설 수립 → 노트북 제작 → GitHub 푸시
2. 사용자: Kaggle에서 노트북 실행 (GPU T4 x2, Internet On) → 출력 숫자/에러를 채팅에 붙여넣기
3. Claude: 결과 분석 → EXPERIMENTS.md·report.html 갱신 → 다음 가설 반영한 노트북 수정
4. 좋은 결과만 리더보드 제출 (description: `expNN | 방법 | local XX%`)

## 확정 전략 (2026-08-04, 사용자 승인)

**하이브리드**: 추론·평가·제출 = Kaggle (무료, 최종 추론 재현성도 Kaggle 환경 기준) / RFT 데이터 생성·QLoRA 학습 = AWS (9시간 벽·T4 속도 병목 회피, 스팟 총 $30~80 예상). 어댑터는 AWS→Kaggle로 가져와 추론.

## 현재 상태 (2026-08-06 밤 갱신) — exp15 완료(H18 두단계 자기수정 기각), 트랙 A·B 사이클 전부 종료, 큐 비어있음

- **exp15 완료(트랙 B3, H18 기각)**: `remote/eval_two_pass.py` — 1차 greedy → 2차 검토·수정(같은 모델), 검증 483문항. 1차 68.9%(333/483), 2차 69.6%(336/483, 답 변경 37건). 비교기준 exp05 greedy 69.4% 대비 +0.2%p로 판정 기준(±1%p) 이내 노이즈 — 개선 없음, H18 기각. 이로써 **pivot-plan.md 트랙 B(B1 exp11·B2 exp14·B3 exp15) 전 항목이 채택 기준(+0.5%p) 미달**로 종료. 트랙 A(exp13에서 완전 동결)와 합쳐 pivot-plan.md 사이클 전체가 유의미한 개선 없이 마무리됨. 현재 최선의 스택은 여전히 exp05/07의 SC n=8~16(74.7~75.6%) — 리더보드 제출은 이 구성 유지 권장. **큐 비어있음** — 로컬 Claude(계획자)가 다음 방향(H8·H9 운영진 답변 확인 또는 새 근본 가설)을 문서화해 큐에 등록 필요
- **exp14 완료(트랙 B2, H19 기각)**: `remote/eval_prompt_ensemble.py` — 프롬프트 4종 × 2샘플 = 8표 다수결, 검증 483문항. 결과 74.3%(359/483), 단일 프롬프트 SC n8(exp05, 74.7%) 대비 -0.4%p로 판정 기준(±1%p 이내는 노이즈) 미달 — 유의미한 개선 없음, H19 기각. 큐 다음 항목: exp15-two-pass(H18, 2단계 자기수정)
- **exp13 완료(트랙 A 완전 동결)**: pivot-plan.md 성공/철수 기준("파일럿 +1%p 미만이면 학습 트랙 동결")을 exp12 결과(SC n8 74.5%, 베이스 대비 -0.2%p, 성공기준 75.7% 미달)에 대입 — **트랙 A(학습 트랙) 완전 동결** 확정. QLoRA SFT 3연속 실패(exp06/06c/09b) + GRPO 파일럿(exp12) 모두 목표 미달로, 시도한 모든 학습 파라다임이 베이스 대비 개선 없음. 향후 자원은 트랙 B(H18 2단계 자기수정, H19 프롬프트 앙상블 SC)에 집중. H8(멀티LoRA)·H9(검증자)는 운영진 답변 대기 지속
- **큐 다음 항목**: 없음 — 로컬 Claude(계획자)가 트랙 B 다음 실험(H18 또는 H19)을 문서화해 큐에 등록해야 함
- **exp12 완료(목표 미달)**: 트랙 A GRPO 파일럿 — `remote/train_grpo.py` 수정 없이 그대로 실행(3000문제, 400스텝, lr5e-6), train_loss 0.02028([wandb ctiym4ny](https://wandb.ai/loonaticvibe2-11-jin-jason/huggingface/runs/ctiym4ny)). 평가: greedy 69.2%(-0.2%p), SC n8 74.5%(-0.2%p, 성공기준 75.7% 미달), 가중투표 74.9%(+0.2%p) — 베이스(69.4/74.7)와 사실상 동일(KL 0.003~0.005로 베이스 근처 유지 확인) 개선 신호 없음

- **exp11 완료(목표 미달)**: 트랙 B1 최적 추론 스택 확정 — SC n=16 다수결 74.7%(361/483), 가중투표 74.3%(359/483). 가중투표가 오히려 다수결보다 낮게 나와 목표(76%+) 미달. exp10(n8)에서는 가중투표가 +0.8%p였는데 부호가 반전돼 **가중투표 효과가 표본·n에 따라 불안정**함을 시사 — 복합 스택 채택 보류, 제출은 SC n=8 유지 권장. n16 다수결 자체도 exp07(75.6%)보다 낮아 vLLM 연속배칭 비결정성(±1%p)도 재확인. 상세는 EXPERIMENTS.md 실험 11 참고. 평가는 이전 세션(중간에 git pull 충돌로 한 차례 중단·자동커밋 후 재개)에 이미 완료돼 있었고 로그(`eval11.log`)·결과 파일(`results/eval_base.json`)로 검증 후 기록만 대행(재실행하지 않음)
- **큐 다음 항목**: exp12-grpo-pilot (트랙 A, GRPO 강화학습 파일럿) — `remote/train_grpo.py`, 수 시간 소요 예상, 성공 기준 SC n8 ≥ 75.7%
- 오늘 추가된 자가회복 장치: 부팅 자동시작(user-data boothook), 불사 러너(set -e 제거), watchdog 실작업 기준(9f3ac4c), ntfy 웹 API로 SSH 없이 서버 관찰 가능

## 이전 상태 (2026-08-06 저녁) — exp06c 완료(재실패), exp08 스킵

- **exp06c QLoRA 재시도도 실패**: greedy 67.9%/SC 73.3% — 베이스(69.4/74.7)보다 하락, exp06(68.3/73.5)과 거의 동일. lr을 1/4로, epoch을 절반으로, 데이터를 문제당 최단 풀이 1개로 줄여도 회복 안 됨 → **학습률 문제가 주 원인이 아니었을 가능성 높음**. 남은 용의선: 자기증류(출력 다양성 감소)·이미 푸는 문제만 학습(데이터 편향). **정체 카운트 2/3** — QLoRA SFT류 다음 한 번 더 실패하면 근본 전환(GRPO/외부데이터 재구성/DPO) 착수
- **exp08 스킵**: 반복 RFT는 exp06c가 베이스보다 좋을 때만 실행하는 조건이었는데 미충족 → 재생성·재학습 실행 안 함
- **exp06 QLoRA 1차 실패**: greedy 68.3%/SC 73.5% — 베이스(69.4/74.7)보다 하락. 원인 가설: lr 과함·자기증류(다양성 감소)·풀 수 있는 문제만 학습
- **exp06b 완료**: 베이스 모델(어댑터 없음) SC n=8로 `results/submission_base.csv` 생성(831행, 정수만, 결측 없음). 리더보드 라벨이 없어 로컬 정확도 산출 불가 — 실점수는 사용자가 Kaggle 제출 후 확인
- **exp07 완료**: SC 스케일링 — n=4 72.7%, n=8 74.7%(exp05), n=16 75.6%. 수확 체감 확인, 제출은 n=8 유지 권장(비용 대비)
- **큐 비어있음** — QLoRA(H3) 계열로는 2연속 실패. 로컬 Claude가 다음 실험(H12 외부 데이터 혼합 또는 H15 DPO 권장)을 큐에 등록해야 함
- 러너 사망 사고 수리: git pull 충돌 시 자동 커밋 후 재시도. 서버 로컬 변경(eval_vllm max_model_len=4096)은 서버에서 커밋 처리
- exp06 학습 자체는 정상(wandb run o5zlvcyp — 학습 로그 제출물 확보됨). 어댑터: 서버 outputs/qlora/qlora_r16_lr0.0002_ep2_final. exp06c 어댑터: outputs/qlora_gentle/qlora_r16_lr5e-05_ep1_final (둘 다 삭제하지 않음)

## 이전 상태 (2026-08-06 오전) — 첫 사이클 결과

- **신기준 기준선 (exp05, 유효 검증 483문항)**: greedy **69.4%** / SC n=8 **74.7%** — H2(SC) +5.3%p 확인. 목표 85%까지 10.3%p
- **RFT 데이터 완성 (exp04·04b)**: 커버리지 79.9%(13,190문제), 필터 후 풀이 36,144개 → data/sft.jsonl (서버에만 존재)
- **exp06 QLoRA 학습 진행 중** → 완료 시 exp07(SC 스케일)·exp08(반복 RFT) 자동 연쇄
- pass@6=79.9% vs SC=74.7% 간극 주목: 다수결이 놓치는 ~5%p는 검증자(H9)·가중 투표(H16)가 노릴 영역
- 참고: 집 IP 바뀌면 SSH 막힘 → SG에 현재 IP authorize (2026-08-06 1회 발생·처리)

## 이전 상태 (2026-08-05) — AWS 자율 루프 가동

- **서버 라이브**: i-006cb0ac24f1ba3b2 (g5.xlarge), 러너가 exp04(RFT 생성)부터 자율 실행 중. watchdog cron 등록됨(유휴 30분 → 자동 stop). IP는 재시작마다 바뀜 → `aws\start.ps1`이 출력
- 서버 세팅 내역: 토큰은 `~/.ajudl_env`(GitHub classic PAT — JinVibe를 collaborator로 초대해 발급, Claude 구독 토큰, wandb 키, NTFY_TOPIC=ajudl-rvcmx3ae). vLLM 0.26.0 + torch 2.11 + unsloth 확인. git identity `ajudl-server`, core.fileMode false
- 서버 접속: `ssh -i ~/.ssh/ajudl-gpu.pem ubuntu@<IP>` / 진행 확인: `tail -f ~/work/deep-learning-challenge/runner.log`
- 결과 수신: ntfy.sh/ajudl-rvcmx3ae (폰) + 서버가 EXPERIMENTS.md·queue.md를 커밋/푸시 → 로컬은 git pull로 동기화. **로컬에서 수정 후엔 반드시 push** (서버가 매 실험 전 pull)
- 보안 메모: 사용자가 토큰들을 채팅에 붙여넣은 적 있음(2026-08-05) → 대회 후 AWS 키·PAT·Claude 토큰·wandb 키 로테이션 권장

## 이전 상태 (2026-08-04)

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

- **추론 시 코드 실행·툴 호출 금지 확정** (운영진 Q&A, 2026-08) — H5(TIR) 폐기. SC/다수결은 허용. 멀티 LoRA 앙상블(H8)·검증자 어댑터(H9)는 질의 중
- **데이터 검수 공지(08-03) 반영**: train 오류 627문항(`deep-learning-challenge-2026/train_filtered_ids.csv`) 학습·검증 제외 (generate_rft·eval_vllm에 반영, exp04는 사후 필터 = 큐의 exp04b). 리더보드는 `deep_chal_math_leaderboard_filtered.csv`(831문항) 사용 — **아직 다운로드 필요**, Kaggle 노트북 재개 시 파일명 교체 필수. 기존 제출은 재제출해야 새 기준 점수

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
