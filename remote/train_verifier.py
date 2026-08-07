"""검증자 v2 어댑터 학습 (H9 재도전) — 풀이를 재검산 근거+Verdict Yes/No로 판정하는 LoRA.

v1(exp18b)과 달리 assistant 응답이 즉답 Yes/No가 아니라 gen_verifier_v2_data.py가
rejection sampling으로 생성한 근거(재검산 몇 줄)+"Verdict: Yes/No" 전문이다.
주의: 생성기 SFT와 달리 '새 기능(채점)'을 얹는 학습이라 기존 조율 파괴 우려가 다름.
추론 시 생성은 베이스(무어댑터), 채점만 이 어댑터를 쓴다.
사용: nohup uv run python remote/train_verifier.py > train_verifier.log 2>&1 &
"""
import json
import os
import random

CONFIG = dict(
    data='data/verifier_v2.jsonl', output_dir='outputs/verifier_v2',
    max_per_class=12000,   # 클래스 균형 1:1
    max_seq_len=2048, lora_r=16, lora_alpha=32,
    lr=1e-4, epochs=1, per_device_batch=4, grad_accum=4, seed=42,
)

VERIFY_PROMPT = (
    'You are a strict math solution grader. Re-derive the key steps of the proposed '
    'solution to check whether its final answer is correct. Be brief (a few lines). '
    'End with exactly one line: "Verdict: Yes" if the final answer is correct, '
    'or "Verdict: No" if it is not.')


def build_samples():
    random.seed(CONFIG['seed'])
    pos, neg = [], []
    for line in open(CONFIG['data'], encoding='utf-8'):
        r = json.loads(line)
        (pos if r['label'] else neg).append(r)
    k = min(len(pos), len(neg), CONFIG['max_per_class'])
    random.shuffle(pos)
    random.shuffle(neg)
    rows = pos[:k] + neg[:k]
    random.shuffle(rows)
    print(f'학습 샘플: 양성 {k} + 음성 {k} = {2*k}')
    return rows


def main():
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='Qwen/Qwen2.5-3B-Instruct',
        max_seq_length=CONFIG['max_seq_len'], load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=CONFIG['lora_r'], lora_alpha=CONFIG['lora_alpha'], lora_dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        random_state=CONFIG['seed'])

    def to_text(r):
        msgs = [{'role': 'system', 'content': VERIFY_PROMPT},
                {'role': 'user', 'content': 'Problem:' + chr(10) + r['question'] +
                 chr(10) * 2 + 'Proposed solution:' + chr(10) + r['solution'][:4000]},
                {'role': 'assistant', 'content': r['judge']}]
        return tokenizer.apply_chat_template(msgs, tokenize=False)

    rows = build_samples()
    ds = Dataset.from_dict({'text': [to_text(r) for r in rows]})

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            output_dir=CONFIG['output_dir'], run_name='verifier_v2_r16_lr1e-4',
            report_to='wandb', dataset_text_field='text',
            max_seq_length=CONFIG['max_seq_len'],
            per_device_train_batch_size=CONFIG['per_device_batch'],
            gradient_accumulation_steps=CONFIG['grad_accum'],
            num_train_epochs=CONFIG['epochs'], learning_rate=CONFIG['lr'],
            lr_scheduler_type='cosine', warmup_ratio=0.03,
            logging_steps=10, save_steps=300, save_total_limit=2,
            seed=CONFIG['seed'], bf16=True))
    trainer.train()

    final = os.path.join(CONFIG['output_dir'], 'verifier_final')
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print('저장:', final)


if __name__ == '__main__':
    main()
