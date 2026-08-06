"""QLoRA SFT 학습 스크립트 (Unsloth) — AWS GPU 서버에서 실행.

사용법 (서버의 tmux 안에서):
    cd ~/work/deep-learning-challenge
    nohup uv run python remote/train_qlora.py > train.log 2>&1 &

데이터: data/sft.jsonl — {"messages": [{"role": ..., "content": ...}, ...]} 형식.
  (RFT 데이터 생성 스크립트가 만들 예정. 검증 세트 val_ids는 반드시 제외돼 있어야 함)
하이퍼파라미터는 아래 CONFIG에서만 바꾼다 — 실험마다 wandb run 이름에 반영된다.
"""
import json
import os

CONFIG = dict(
    base_model='Qwen/Qwen2.5-3B-Instruct',  # 대회 규칙: 고정
    data_path='data/sft_short.jsonl',
    output_dir='outputs/qlora_gentle',
    max_seq_len=2048,
    lora_r=16,
    lora_alpha=32,
    learning_rate=5e-5,
    epochs=1,
    per_device_batch=4,
    grad_accum=4,
    seed=42,
)

RUN_NAME = f"qlora_r{CONFIG['lora_r']}_lr{CONFIG['learning_rate']}_ep{CONFIG['epochs']}"


def main():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG['base_model'],
        max_seq_length=CONFIG['max_seq_len'],
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        lora_dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        random_state=CONFIG['seed'],
    )
    tokenizer = get_chat_template(tokenizer, chat_template='qwen-2.5')

    dataset = load_dataset('json', data_files=CONFIG['data_path'], split='train')

    def to_text(ex):
        return {'text': tokenizer.apply_chat_template(
            ex['messages'], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(to_text)
    print(f'학습 샘플 {len(dataset)}개, 예시:\n{dataset[0]["text"][:500]}')

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=CONFIG['output_dir'],
            run_name=RUN_NAME,
            report_to='wandb',                    # 학습 로그 = 검증 제출물 권장 항목
            dataset_text_field='text',
            max_seq_length=CONFIG['max_seq_len'],
            per_device_train_batch_size=CONFIG['per_device_batch'],
            gradient_accumulation_steps=CONFIG['grad_accum'],
            num_train_epochs=CONFIG['epochs'],
            learning_rate=CONFIG['learning_rate'],
            lr_scheduler_type='cosine',
            warmup_ratio=0.03,
            logging_steps=10,
            save_steps=200,                       # 스팟 회수 대비 주기 저장
            save_total_limit=2,
            seed=CONFIG['seed'],
            bf16=True,                            # A10G는 bf16 지원 (T4면 fp16으로)
        ),
    )
    trainer.train(resume_from_checkpoint=any(
        d.startswith('checkpoint-') for d in
        (os.listdir(CONFIG['output_dir']) if os.path.isdir(CONFIG['output_dir']) else [])))

    final_dir = os.path.join(CONFIG['output_dir'], RUN_NAME + '_final')
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, 'config_used.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f'저장 완료: {final_dir}')


if __name__ == '__main__':
    main()
