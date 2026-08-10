# exp30 — 최종 test 실전 리허설 보고서 (2026-08-10)

**목적**: 8/31 최종 test(2,000문항)를 하루 안에 무사고로 처리하기 위한 운영 리스크 점검. 리더보드 831문항을 최종 test의 대역으로 삼아 확정 스택(가중 SC n=32, temp 0.7) 전 과정을 시간·자원 실측하며 1회 완주. 성능 실험이 아니므로 성공/실패 판정 없음.

## 실행 정보

- **명령**: `uv run python remote/make_submission.py --n 32 --weighted --tag rehearsal --lb-csv deep-learning-challenge-2026/deep_chal_math_leaderboard_filtered.csv`
- **대상**: 리더보드 831문항 전체 (`deep_chal_math_leaderboard_filtered.csv`)
- **모델**: Qwen2.5-3B-Instruct(vLLM), 어댑터 없음, temperature=0.7, top_p=0.8, max_tokens=2048, 확신도 가중 다수결, n=32
- **실행 구간**: 2026-08-10T15:49:05Z ~ 2026-08-10T16:50:36Z
- **GPU 메모리 추적**: `nvidia-smi`를 10초 간격 폴링해 `gpu_mem_rehearsal.csv`에 기록(별도 프로세스, 관측 전용 — 시간 측정에 영향 없음)

## 결과 수치

| 항목 | 값 |
|---|---|
| 처리 문항 수 | 831/831 (전 문항 완주, 빈 답 없음) |
| 총 소요 시간(wall clock) | 1:01:30 (3,690초) |
| **문항당 처리 시간** | **4.44초/문항** |
| **2,000문항 환산 시간** | **8,881초 ≈ 2시간 28분** |
| GPU 메모리 최대치 | 21,891 MiB / 23,028 MiB 총량 (**약 95.1%**, A10G 24GB) |
| CPU RSS(프로세스 상주 메모리) 최대치 | 5,315,340 KB ≈ **5.07 GiB** |
| CPU 사용률(`/usr/bin/time` 기준) | 37% (User 1338.7s + System 46.7s / 경과 3690s) — **GPU 바운드**, CPU 여유 충분 |
| 오류·경고 | 없음 (JIT 컴파일 warm-up 로그 외 이상 없음) |
| 결과 파일 검증 | `results/submission_rehearsal.csv` 832행(헤더+831), id 831개 전부 유일, answer 전부 정수, 결측 0건 |

## 병목 지점

- **GPU 연산 자체가 병목** (CPU 사용률 37%로 낮음 — I/O나 전처리 대기가 아니라 vLLM 생성이 시간 대부분을 차지). 문항당 32샘플 생성(각 max_tokens=2048)이 지배적 비용.
- `remote/make_submission.py`는 100문항 단위 청크로 처리하지만(exp29 OOM 사고 후 도입) 이는 **메모리 절약용**일 뿐 진행 상황을 디스크에 저장하지 않는다 — 청크가 끝나도 `preds` 리스트는 프로세스 메모리에만 쌓이고, CSV는 전체 831문항이 끝난 뒤 단 한 번 기록된다.

## 중단 시 재개 가능 여부 — ⚠️ 불가능 (가장 중요한 발견)

- `make_submission.py`는 **체크포인트/재개 로직이 없다**. 831문항 처리 중 830문항째에서 프로세스가 죽어도 저장된 결과는 0건이며, 재시작 시 **처음부터 전체를 다시 실행**해야 한다.
- 반면 같은 서버에 있는 `remote/dump_lb_samples.py`는 100문항 청크마다 `results/lb_samples.jsonl`에 append하고 `.progress` 파일에 완료된 id 목록을 기록해 **중단 후 재개가 실제로 동작**한다(exp33에서 검증됨). `make_submission_from_dump.py`는 이 덤프에서 GPU 없이 수 초 만에 제출 파일을 뽑아낸다.
- **결론**: 2,000문항을 직접 `make_submission.py`로 돌리는 것은 8/31 당일 최대 리스크(중간 사망 시 전량 재실행, ~2.5시간 손실)를 그대로 안고 가는 방식이다. **`dump_lb_samples.py`(재개 가능) + `make_submission_from_dump.py`(집계, GPU 불필요) 조합으로 대체하는 것을 강력히 권장**한다.

## 8/31 당일 실행 체크리스트 (초안)

1. **사전(전날)**: 서버 부팅·GPU 정상 확인, `git pull`로 최신 스크립트 확보, `deep-learning-challenge-2026/`에 최종 test 파일(2,000문항) 배치 확인, 디스크 여유 공간 확인(로그+jsonl 덤프 대비 최소 1GB)
2. **실행은 반드시 dump 방식 사용**: `nohup uv run python remote/dump_lb_samples.py --n 32 --lb-csv <최종test경로> --out results/final_samples.jsonl > dump_final.log 2>&1 &` (n=32는 로컬 최고 스택 기준. 시간 여유가 있으면 n=48도 고려 가능 — 2,000문항 기준 예상 시간은 아래 참고)
3. **모니터링**: 5~10분 간격으로 로그의 "진행 N/2000문항" 라인 확인. `gpu_mem_rehearsal.csv`처럼 `nvidia-smi` 폴링을 병행해 메모리 이상 조기 감지
4. **예상 소요 시간**: 이번 리허설(n=32, 831문항, 4.44초/문항)을 그대로 환산하면 2,000문항 n=32 기준 **약 2시간 28분**. dump 방식은 make_submission.py와 동일한 생성 로직이라 시간대는 동일하게 적용 가능
5. **중단 시**: 프로세스가 죽어도 `results/final_samples.jsonl` + `.progress`가 남아있으므로 **같은 명령을 그대로 재실행**하면 완료된 문항을 건너뛰고 이어서 진행됨 (재확인: 재실행 전 `.progress`의 `done` 개수와 `.jsonl` 줄 수가 일치하는지 확인)
6. **완료 후**: `uv run python remote/make_submission_from_dump.py --dump results/final_samples.jsonl --rule weighted --n 32 --tag final --lb-csv <최종test경로>` (수 초, GPU 불필요) → `results/submission_final.csv` 생성
7. **제출 전 검증**: 행수(=문항 수), 컬럼(`id`,`answer`), `answer` 전부 정수, 결측·중복 없음을 스크립트로 재확인 (이번 리허설과 동일한 방식)
8. **최종 test 제출은 사용자 확인 후에만** — 자동 제출 금지 (CONTEXT.md 원칙)
9. **여유 시간 확보**: 2.5시간 예상 + 재시도 여유(중단 1회 가정 시 최대 5시간)를 감안해 마감 최소 6시간 전에는 착수 시작

## GPU 메모리 시계열 참고

- 폴링 시작 시 0~19 MiB(모델 로드 전) → 로드 직후 급증 → 처리 중 7,359~21,891 MiB 사이에서 변동 → 처리 종료 후 0 MiB로 복귀(정상 해제 확인)
- 원본 데이터: `gpu_mem_rehearsal.csv` (10초 간격, 398행)
