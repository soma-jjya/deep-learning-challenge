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

## 현재 상태 (2026-08-07 갱신 3) — exp19 완료(정책 앙상블, H8 기각), 큐 다음은 exp20(퓨샷)

- **exp19 완료(정책 앙상블, H8 기각)**: `remote/eval_policy_ensemble.py` — 세션 진입 시 이미 이전 세션이 백그라운드로 실행·완료해둔 결과(02:08 완료, 스크립트·결과 파일 존재, 프로세스 종료됨)를 발견해 재실행 없이 검증 후 기록만 수행. base8+qlora8(exp06 어댑터) 16표 합동 다수결 74.9%(362/483), base8+grpo8(exp12 어댑터) 74.7%(361/483) — 둘 다 균일 n16(exp17, 75.6%)보다 낮고 성공 기준(76.6%+) 미달. 어댑터 자체가 베이스 대비 우위가 없는 상태(트랙 A 동결 사유)라 정책 다양성 이득의 전제가 없었던 것으로 해석. 총력전 사이클(exp19~22) 4장 중 1장 소진(H8 기각)
- **큐 다음 항목**: exp20-fewshot (4-shot 프롬프트, `data/sft_short.jsonl`에서 유형별 짧은 풀이 4개 선정해 예시로 붙이고 greedy/SC n8 평가, 성공 기준 SC≥75.7%)
- **운영 메모(러너, 지속 이월)**: `remote/run_experiments.sh` 러너가 재부팅(`=== boot restart ===`, runner.log) 후 exp19를 다시 실행 트리거했으나 이미 완료돼 있던 결과였음(중복 실행은 없었음). 다음 안전한 시점에 러너 상태 재확인 권장

## 이전 상태 (2026-08-07 갱신 2) — exp18c 완료(Best-of-N 평가, H9 기각), 큐 비어있음

- **exp18c 완료(검증자 Best-of-N 평가, H9 최종 검증)**: `remote/eval_bestofn.py --verifier outputs/verifier/verifier_final --n 8` — 세션 진입 시 이미 백그라운드에서 실행 중이던 프로세스(01:13 시작, PID 17615)를 발견해 중복 실행 없이 완료까지 대기(약 14분). 결과: majority 74.9%(362/483), verifier_weighted **75.4%**(364/483), best_of_1 72.5%(350/483). **성공 기준(verifier_weighted≥76.5%) 미달** — H9(검증자 Best-of-N) 기각. verifier_weighted가 majority보다는 +0.5%p 높아 방향은 맞으나 목표치에 크게 못 미침. best_of_1(검증자 단독 최고점 선택)은 오히려 -2.4%p 낮음. exp18d(제출 파일)는 조건 미충족으로 skip 기록
- **H9까지 종료로 안전한 가설 풀 소진**: 트랙 A(학습, QLoRA/GRPO)·트랙 B(추론 재설계, 자기수정/프롬프트앙상블/가중투표)·H9(검증자) 전부 목표 미달로 종료. **큐 비어있음 — 로컬 Claude(계획자)가 다음 근본 가설(H15 DPO 또는 H13b 검증셋 라벨 재검수)을 문서화해 큐에 등록 필요**. CONTEXT.md 운영 정책(가설 풀 3개 이상 유지)상 시급
- **운영 메모(러너, 미해결 이월)**: `remote/run_experiments.sh` 러너 프로세스(PID 1165, 08-06 17:28부터 실행 중)가 최근 pgrep 패턴 수정 커밋(fe674f0)을 반영 못한 채 옛 로직으로 동작 중인 것으로 추정 — 세션들이 진행 중인 프로세스를 감지하고 대기만 하는 패턴이 계속 반복됨(중복 실행은 없었음, 낭비는 재호출 오버헤드 수준). 이 세션도 부모 프로세스라 직접 kill하지 않음. **큐가 빈 지금이 안전하게 재시작할 최적 시점** — 다음 세션(또는 사용자)이 큐 새 항목 등록 후 러너가 자연 종료·재기동되는지 확인 권장. 자연 종료가 안 되면 별도 세션에서 안전하게 재시작 검토

## 이전 상태 (2026-08-06 밤 갱신 3) — exp18a 완료(검증자 데이터 생성), 큐 다음은 exp18b(검증자 학습)

- **exp18a 완료(검증자 데이터 생성, H9)**: `remote/gen_verifier_data.py` — 베이스 모델로 학습·검증 제외 6,000문제를 문제당 4샘플씩 생성(temp 0.8, top_p 0.95), 답 일치로 자동 라벨링. 6,000/6,000 처리, 총 24,000쌍 → `data/verifier.jsonl`. **양성(정답) 16,240개(67.7%) / 음성(오답) 7,760개(32.3%)** — 약 2.1:1 불균형. exp18b는 큐 명세대로 1:1 클래스 균형 샘플링 예정이라 음성 7,760개가 상한(균형 데이터 최대 ~15,520개). 실행 자체는 예상(~5시간)보다 훨씬 빨리 끝남(약 1시간, 22:44 모델 로드 → 23:43 완료) — vLLM 배치 처리가 예상보다 효율적이었던 것으로 보임
- **큐 다음 항목**: exp18b-verifier-train (검증자 어댑터 학습, `remote/train_verifier.py`, 클래스 균형 1:1, ~1-2시간) → exp18c(Best-of-N 평가, 성공 기준 verifier_weighted≥76.5%) → exp18d(성공 시 제출)

## 이전 상태 (2026-08-06 밤 갱신 2) — exp17/17b 완료(SC n32/n64, H11 종료), 큐 다음은 exp18a(검증자 데이터 생성)

- **exp17 완료(SC 스케일 마무리, H11 종료)**: `remote/eval_vllm.py --mode sc --n 32`/`--n 64` (베이스, 검증 483문항). n32 다수결 75.4%(364/483)/가중 75.8%(366/483), n64 다수결 75.6%(365/483)/가중 75.4%(364/483). **성공 기준 76.3%+ 미달** — n=16(75.6%) 이후로는 표본을 4배 늘려도 74.7~75.8% 사이에서 정체·비단조. H11(SC 스케일링) 가설 완전 종료: n=8~16이 비용 대비 최적, 그 이상은 이득 없음. exp17b(제출 파일 생성)는 성공 기준 미달로 **skip**, 기존 submission_base.csv(SC n=8, LB 0.77015) 유지. **운영 메모**: 이번 세션 시작 시 발견한 바로는, 이전 여러 세션이 연속으로 정지·재시작되며 exp17을 큐에 체크하지 못한 채 n=32 평가를 최소 4차례 재실행(약 4~5시간 GPU 낭비, runner.log에 기록)했다 — eval_vllm.py가 결과를 n 구분 없이 항상 `results/eval_base.json`에 덮어쓰는 구조라 재개 시 "이미 완료됐는지" 판단이 어려웠던 것이 원인으로 보임. 이번 세션은 재실행 없이 로그(eval17_n32.log, eval17_n64.log)로 확인 가능한 마지막 결과만 채택해 기록 완료. **개선 제안(다음 세션 참고)**: eval_vllm.py 출력 파일명에 `--n`을 포함시키면(예: eval_base_n{n}.json) 이런 재실행 낭비를 막을 수 있음 — 아직 미반영
- **큐 다음 항목**: exp18a-verifier-data (H9, 검증자 데이터 생성 — 운영진 허용 확정, `remote/gen_verifier_data.py`, ~5시간) → exp18b(검증자 학습) → exp18c(Best-of-N 평가, 성공 기준 verifier_weighted≥76.5%) → exp18d(성공 시 제출)로 이어지는 체인. 남은 유효 가설은 사실상 H9(검증자)와 H13b(검증셋 라벨 재검수, 미등록)뿐 — SC 스케일링·프롬프트 앙상블·자기수정·가중투표·QLoRA SFT·GRPO는 모두 시도 완료(트랙 A 동결, 트랙 B 전 항목 기각, H11 종료)

## 이전 상태 (2026-08-06 밤 갱신) — exp16 완료(오답 정밀 분석), 큐 다시 비어있음

- **exp16 완료(오답 정밀 분석, H13 사전 단계)**: `remote/eval_error_analysis.py` — SC n=8 재실행(베이스, 검증 483문항), 73.7%(356/483), 오답 127건을 `results/wrong_analysis.jsonl`에 기록. 오답 중 30건은 `remote/eval_error_sample_read.py`(보조 스크립트, greedy 전체 텍스트 확보)로 직접 읽고 4유형 분류: **①답 추출 실패 0건, ②계산 실수 3건(10%), ③접근 자체 오류 20건(67%), ④정답 라벨 의심 7건(23%)**. 핵심 시사점: (a) ①이 0건이라 후처리/추출 로직 개선으로 얻을 공짜 점수는 없음 — 지금까지 SFT/GRPO가 모두 실패한 것과 일관되게, 문제는 얕은 후처리가 아니라 추론 능력 자체(③이 압도적). (b) ④가 23%로 상당해 검증 세트(483문항) 라벨 신뢰도에 의문 — 실측 정확도(74.7% 안팎)의 해석에 노이즈로 남을 수 있음. H13 본체(표적 데이터 증강)는 트랙 A가 동결된 상태라 보류. **큐 비어있음** — 로컬 Claude(계획자)가 다음 방향을 정해야 함. 후보: 검증 세트 라벨 재검수(H13b, ④ 23% 근거), H8·H9 운영진 답변 확인, 또는 새 근본 가설
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
