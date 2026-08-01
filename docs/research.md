# 리서치 노트 — 유사 사례와 근거 (2026-08-01)

목표 85%를 향한 각 가설의 근거 자료. report.html의 참고 문헌 원본.

## 1. 우리와 같은 환경(T4 x2)에서 수학 대회를 우승한 사례

**NuminaMath — Kaggle AIMO Progress Prize 1위** ([HF 블로그](https://huggingface.co/blog/winning-aimo-progress-prize), [GitHub](https://github.com/project-numina/aimo-progress-prize))
- Kaggle T4 x2 환경에서 vLLM으로 추론. T4는 bf16·Flash Attention 2 미지원이라 **GPTQ 8bit 양자화**로 속도 확보
- **SC-TIR**: 같은 문제를 N번 샘플링 + 모델이 파이썬 코드를 쓰면 실행해서 결과를 다시 넣어주고(Tool-Integrated Reasoning), 최종 다수결
- 시사점: ① SC(다수결)는 검증된 대회 필승 전략 ② 추론 속도가 병목이면 양자화+vLLM ③ 파이썬 실행은 오프라인이라 대회 규칙에도 합치

**AIMO-2 우승 솔루션** ([arXiv:2504.16891](https://arxiv.org/pdf/2504.16891)), [AIMO-3 관련 분석](https://arxiv.org/pdf/2603.27844) — 추론 시점 최적화보다 **모델 자체 능력(학습)이 지배적**이라는 결론. SC로 몇 %p 얻은 뒤엔 결국 SFT가 본게임

## 2. Self-Consistency (다수결)

- 원 논문: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning" (ICLR 2023, arXiv:2203.11171) — GSM8K에서 +17.9%p 사례
- [AIMO-3 상위권 writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/entropy-weighted-self-consistency-for-olympiad-lev): 엔트로피 가중 다수결 변형
- 기대: 정답이 "가끔" 나오는 문제를 다수결이 건져줌. 소형 모델일수록 효과 큼

## 3. Rejection Sampling Fine-Tuning (RFT/STaR) — SFT 데이터 만들기

- [RFT 논문 (arXiv:2308.01825)](https://arxiv.org/abs/2308.01825) "Scaling Relationship on Learning Mathematical Reasoning": 모델이 스스로 여러 풀이를 생성 → **정답에 도달한 풀이만** 골라 SFT. 소형 모델·저자원 환경에서 GSM8K 대폭 향상
- STaR (arXiv:2203.14465), [개념 정리](https://www.emergentmind.com/topics/rejection-sampling-fine-tuning-rsft)
- 핵심 변수: **서로 다른(distinct) 풀이 경로의 개수**가 성능을 좌우 → 샘플 수를 늘리는 게 유리
- 우리 적용: train 17k에 답만 있으므로, 베이스 모델로 문제당 4~8개 풀이 생성 → 정답 것만 SFT 데이터화

## 4. 공개 CoT 데이터셋 (외부 데이터 — 무료 공개라 규칙 허용)

- [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT): 경시 문제 86만 쌍, Apache 2.0. 우리 데이터의 LaTeX 경시형(48%)과 유사
- [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2): 1,400만 쌍 (Llama3.1-405B 생성)
- 사용 시 README 사용 데이터 목록에 기재 필수

## 5. 베이스 모델 성능 참고치

- [Qwen2.5-3B-Instruct 정리](https://www.emergentmind.com/topics/qwen2-5-3b-instruct-model): GSM8K(초등~중등 서술형) zero-shot ~83%. 우리 66%는 데이터에 경시형이 절반 섞인 탓으로 해석
- [Qwen2.5-Math 블로그](https://qwenlm.github.io/blog/qwen2.5-math/) — 같은 3B급도 수학 특화 학습으로 큰 격차. 단 Qwen2.5-Math 모델 자체는 **베이스 교체 금지 규칙상 사용 불가**, "어떤 학습이 효과 있는가"의 참고로만
- LoRA로 3B급 수학 향상 사례: [LoRR/DPO 실험](https://arxiv.org/pdf/2508.06412) +4.3%p(GSM8K), [소형 모델 증류 연구](https://aclanthology.org/2025.findings-acl.1301.pdf) — 소형 모델은 "너무 강한 교사"의 긴 풀이보다 짧고 단순한 풀이가 잘 맞음 → CoT 데이터 고를 때 풀이 길이 주의

## 종합 → 가설 우선순위에 반영

1. SC는 싸고 확실 (+5~10%p 기대) → 먼저
2. 그다음은 RFT-SFT (자체 생성 CoT + NuminaMath-CoT 혼합)가 본게임
3. TIR(파이썬 실행)은 SFT가 자리잡은 뒤 (base instruct는 코드 신뢰도 낮음)
4. 양자화+vLLM은 속도 병목이 오면 (SC 샘플 수를 늘리기 위한 수단)
