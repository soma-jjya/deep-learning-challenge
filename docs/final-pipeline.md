# 최종 test 실행 절차 (2026-08-31) — 확정본

> exp30 리허설(2026-08-10, 831문항 실측)로 검증된 절차. **당일에는 이 문서만 따른다.**
> 리허설 원본 수치는 `results/rehearsal_report.md`.

## 확정 스택

**Qwen2.5-3B-Instruct (무학습 베이스) + 확신도 가중 Self-Consistency n=32, temperature 0.7, top_p 0.8**

- 어댑터 미사용 — 학습 5전(SFT×3, GRPO×2) 전부 베이스 이하로 확인돼 최종 스택에서 제외
- 근거: 로컬 검증 75.8%, 리더보드 0.78459 (참값은 재실행 변동 감안 시 0.780±0.005)

## ⚠️ 반드시 dump 방식으로 실행할 것

`make_submission.py`를 직접 쓰면 **중단 시 처음부터 재실행**(최대 2.5시간 손실)이다. 리허설에서 확인된 최대 리스크.
아래 2단계 방식은 100문항마다 저장돼 **어디서 죽어도 이어서 재개**된다.

### 1단계 — 표본 생성 (약 2시간 30분, 재개 가능)

```bash
cd ~/work/deep-learning-challenge && git pull
nohup uv run python remote/dump_lb_samples.py \
  --n 32 --temp 0.7 --top-p 0.8 \
  --lb-csv deep-learning-challenge-2026/<최종test파일>.csv \
  --out results/final_samples.jsonl > dump_final.log 2>&1 &
```

- 진행 확인: `tail -f dump_final.log` → "진행 N/2000문항"
- **중단됐다면 같은 명령을 그대로 재실행** (완료분은 건너뜀). 재실행 전 `.progress`의 done 개수와 jsonl 줄 수 일치 확인

### 2단계 — 제출 파일 생성 (수 초, GPU 불필요)

```bash
uv run python remote/make_submission_from_dump.py \
  --dump results/final_samples.jsonl --rule weighted --n 32 \
  --lb-csv deep-learning-challenge-2026/<최종test파일>.csv --tag final
```

### 3단계 — 제출 전 검증 (필수)

- 행수 = 문항 수(2,000) / 컬럼 `id`,`answer` / answer 전부 정수 / 결측·중복 0건
- **`id`는 소문자** (대회 문서엔 `ID`로 적혀 있으나 채점기는 소문자 — 2026-07-31 첫 제출 ERROR로 확인된 사항)

### 4단계 — 제출

```bash
python -m kaggle competitions submit -c deep-learning-challenge-2026 \
  -f results/submission_final.csv -m "final | weighted SC n32 | local 75.8%"
```

**⚠️ 최종 제출은 사용자 확인 후에만. 자동 제출 금지.**

## 시간 계획

| 구간 | 소요 | 비고 |
|---|---|---|
| 표본 생성 | 2시간 30분 | 4.44초/문항 × 2,000 |
| 집계·검증 | 5분 | GPU 불필요 |
| **중단 1회 대비 여유** | +2시간 30분 | 재개되므로 실제로는 부분 손실만 |
| **권장 착수 시각** | **마감 6시간 전** | 최악의 경우도 흡수 |

## 사전 점검 (전날)

- [ ] 인스턴스 기동·GPU 인식 (`nvidia-smi`)
- [ ] `git pull`로 최신 스크립트 확보
- [ ] 최종 test CSV를 `deep-learning-challenge-2026/`에 배치, 컬럼명(`id`,`question`) 확인
- [ ] 디스크 여유 1GB 이상 (덤프 jsonl + 로그)
- [ ] Kaggle 토큰 유효성 확인 (`kaggle competitions submissions -c deep-learning-challenge-2026`)
- [ ] 일일 제출 한도(5회) 잔여 확인 — 최종일엔 재시도 여지를 남길 것

## 사고 대응

| 상황 | 대응 |
|---|---|
| 생성 프로세스 사망 | 같은 명령 재실행 → 완료분 건너뛰고 재개 |
| 러너/서버 무응답 | AWS API로 재부팅 (SSH 불가해도 가능) → systemd가 러너 자동 기동 |
| SSH 차단 | 네트워크 문제 — AWS API·ntfy는 별개로 동작, 재부팅 경로로 우회 |
| GPU 메모리 부족 | `--chunk 50`으로 청크 축소 (리허설 최대 95% 사용, 여유 적음) |
| 시간 부족 | `--n 16`으로 축소 (로컬 75.6% — n32와 0.2%p 차이, 사실상 동등) |
