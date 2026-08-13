"""보상 헤드 학습 (H24) — 문제 내 쌍대 랭킹으로 Best-of-N 재순위화용 스칼라 점수기.

exp18c(검증자 v1)와 무엇이 다른가 — 두 가지가 다르다:

1) **목적함수**. exp18c는 "이 풀이가 맞나?"를 Yes/No **토큰으로 생성**하게 학습시켰다.
   그건 문제 간 절대 채점을 배우는 과제다. 그러나 Best-of-N이 실제로 요구하는 것은
   **같은 문제 안에서 후보끼리의 순위**다. 여기서는 한 문제의 (정답풀이, 오답풀이) 쌍에
   Bradley-Terry 랭킹 손실을 걸어 **그 과제 자체**를 최적화한다.
2) **출력 경로**. 생성 헤드가 아니라 마지막 은닉상태 위의 **스칼라 헤드**를 쓴다.
   "정오를 말할 줄 모른다"와 "정오를 표현에 담지 못한다"는 다른 명제이며,
   우리는 후자를 한 번도 시험한 적이 없다. 운영진이 파라미터 추가를 명시 허용(2026-08-11).

겨냥하는 여지: 상위 2후보를 완벽히 고르면 +4.14%p(exp50 측정치).

데이터: data/verifier.jsonl (6,000문제 x 4샘플 = 24,000건, 정답 라벨 자동 부여).
한 문제 안에서 정답풀이 x 오답풀이의 모든 교차쌍을 만든다(추가 생성 0).

사용:
    uv run python remote/prep_reward_pairs.py
    nohup uv run python remote/train_reward_head.py > train_rm.log 2>&1 &
"""
import json
import os

CONFIG = dict(
    base_model='Qwen/Qwen2.5-3B-Instruct',   # 대회 규칙: 고정
    data_path='data/reward_pairs.jsonl',
    output_dir='outputs/reward_head',
    max_seq_len=1024,          # 문제+풀이. A10G 22GB에 맞춘 값
    lora_r=8,                  # 표현을 조금 움직일 여지만 (헤드만으로는 약할 수 있음)
    lora_alpha=16,
    learning_rate=1e-5,
    epochs=1,
    per_device_batch=1,        # 쌍당 2회 순전파라 실효 배치는 2배
    grad_accum=16,
    seed=42,
)
# 메모리 설계 메모 (2026-08-11, OOM 2회 후 수정):
#  ① AutoModelForCausalLM 대신 AutoModel(= 트랜스포머 본체)만 쓴다.
#     보상 헤드는 은닉상태만 필요한데 LM 헤드는 어휘 151,936차원 로짓과 그 기울기까지
#     만들어 배치당 1GB 가까이 먹는다 — 전혀 쓰지 않는 계산이다.
#  ② 4bit 양자화로 가중치를 6.2GB -> 약 2GB로 낮춘다.
#  ③ gradient checkpointing으로 역전파용 활성값을 줄인다.

RUN_NAME = f"rm_r{CONFIG['lora_r']}_lr{CONFIG['learning_rate']}"

BS = chr(92)
SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.')


def build_text(tok, question, solution):
    """추론 때와 동일한 프롬프트 형식 — 학습·추론 불일치는 그 자체로 성능 손실이다."""
    return tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': question},
         {'role': 'assistant', 'content': solution}],
        tokenize=False, add_generation_prompt=False)


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(CONFIG['base_model'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # AutoModel = LM 헤드 없는 트랜스포머 본체. last_hidden_state만 돌려준다.
    base = AutoModel.from_pretrained(
        CONFIG['base_model'], dtype=torch.bfloat16, device_map='cuda',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True))
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    base = get_peft_model(base, LoraConfig(
        r=CONFIG['lora_r'], lora_alpha=CONFIG['lora_alpha'], lora_dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        task_type='FEATURE_EXTRACTION'))

    hidden = base.config.hidden_size
    head = nn.Linear(hidden, 1, dtype=torch.bfloat16).cuda()
    nn.init.zeros_(head.bias)
    nn.init.normal_(head.weight, std=0.01)

    class Pairs(Dataset):
        def __init__(self, path):
            self.rows = [json.loads(l) for l in open(path, encoding='utf-8')]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            return (build_text(tok, r['question'], r['chosen']),
                    build_text(tok, r['question'], r['rejected']))

    ds = Pairs(CONFIG['data_path'])
    print(f'쌍 {len(ds)}개')

    def score(texts):
        enc = tok(texts, return_tensors='pt', padding=True, truncation=True,
                  max_length=CONFIG['max_seq_len']).to('cuda')
        h = base(**enc).last_hidden_state               # (B, T, H) — 로짓 계산 없음
        # 우패딩이므로 각 시퀀스의 마지막 유효 토큰 위치를 직접 고른다
        idx = enc['attention_mask'].sum(1) - 1
        last = h[torch.arange(h.size(0), device=h.device), idx]   # (B, H)
        return head(last.to(head.weight.dtype)).squeeze(-1)       # (B,)

    params = [p for p in base.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=CONFIG['learning_rate'])
    dl = DataLoader(ds, batch_size=CONFIG['per_device_batch'], shuffle=True)

    step = 0
    run_loss = run_acc = seen = 0.0
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    for ep in range(CONFIG['epochs']):
        for chosen, rejected in dl:
            s = score(list(chosen) + list(rejected))
            b = len(chosen)
            sc, sr = s[:b], s[b:]
            # Bradley-Terry: 정답 풀이가 오답 풀이보다 높은 점수를 받도록
            loss = -torch.nn.functional.logsigmoid(sc - sr).mean()
            (loss / CONFIG['grad_accum']).backward()
            run_loss += loss.item() * b
            run_acc += (sc > sr).float().sum().item()
            seen += b
            step += 1
            if step % CONFIG['grad_accum'] == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
            if step % (CONFIG['grad_accum'] * 10) == 0:
                print(f'  step {step}  loss {run_loss/seen:.4f}  '
                      f'쌍 정확도 {run_acc/seen:.3f}', flush=True)
                run_loss = run_acc = seen = 0.0

    out = os.path.join(CONFIG['output_dir'], RUN_NAME)
    base.save_pretrained(out)
    torch.save(head.state_dict(), os.path.join(out, 'reward_head.pt'))
    json.dump(CONFIG, open(os.path.join(out, 'config_used.json'), 'w'), indent=2)
    print(f'저장 완료: {out}')
    print('다음: uv run python remote/eval_reward_bestofn.py --rm ' + out)


if __name__ == '__main__':
    main()
