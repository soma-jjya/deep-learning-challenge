"""DPO 학습 (H15) — 학습 축에서 유일하게 손대지 않은 방법.

전제: 이 모델은 SFT를 하면 정확도보다 **SC 다양성이 먼저 죽는다**(exp09b). 그래서
- LoRA rank를 SFT 때(16)보다 낮게(8) 잡고
- 학습률·에폭을 작게 두고 (beta를 높여 참조 모델에서 멀어지지 않게 KL을 세게 건다)
- **평가는 반드시 SC n=32로 한다. greedy만 보면 속는다.**

사용:
    uv run python remote/prep_dpo_data.py
    nohup uv run python remote/train_dpo.py > train_dpo.log 2>&1 &
    uv run python remote/eval_vllm.py --mode both --adapter outputs/dpo/<RUN_NAME>_final
"""
import json
import os

CONFIG = dict(
    base_model='Qwen/Qwen2.5-3B-Instruct',  # 대회 규칙: 고정
    data_path='data/dpo.jsonl',
    output_dir='outputs/dpo',
    max_seq_len=2048,
    max_prompt_len=768,
    lora_r=8,           # SFT(16)보다 낮게 — 다양성 붕괴 위험을 줄인다
    lora_alpha=16,
    learning_rate=5e-6,  # DPO 표준 대역. SFT의 1e-4를 그대로 쓰면 붕괴한다
    beta=0.3,            # 클수록 참조 모델에 가깝게 묶임(=다양성 보존). 표준 0.1보다 보수적으로
    epochs=1,
    per_device_batch=2,
    grad_accum=8,
    seed=42,
)

RUN_NAME = f"dpo_r{CONFIG['lora_r']}_b{CONFIG['beta']}_lr{CONFIG['learning_rate']}"

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + chr(92) + 'boxed{}.')


def _patch_transformers_cache():
    """TRL 0.24의 DPOTrainer는 mergekit을 강제 임포트하는데, mergekit이 transformers 5.x에서
    삭제된 상수 TRANSFORMERS_CACHE를 참조해 임포트가 깨진다. mergekit을 실제로 쓰지는 않으므로
    (병합 콜백 전용) 이 프로세스 안에서만 상수를 되살려 우회한다.
    공유 venv의 transformers를 다운그레이드하면 vLLM(=8/31 최종 파이프라인)이 깨지므로 금지.
    """
    import transformers.utils.hub as hub
    if not hasattr(hub, 'TRANSFORMERS_CACHE'):
        hub.TRANSFORMERS_CACHE = os.environ.get(
            'HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))


def main():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset

    _patch_transformers_cache()
    from trl import DPOTrainer, DPOConfig

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

    def to_pref(ex):
        # prompt는 채팅 템플릿을 적용한 문자열, chosen/rejected는 이어질 응답 본문.
        # TRL이 prompt+chosen / prompt+rejected로 붙여 로그확률을 비교한다.
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': ex['question']}],
            tokenize=False, add_generation_prompt=True)
        eos = tokenizer.eos_token or ''
        return {'prompt': prompt,
                'chosen': ex['chosen'] + eos,
                'rejected': ex['rejected'] + eos}

    dataset = dataset.map(to_pref, remove_columns=dataset.column_names)
    print(f'선호쌍 {len(dataset)}개')
    print('--- prompt 예시 ---' + chr(10) + dataset[0]['prompt'][:400])
    print('--- chosen 예시 ---' + chr(10) + dataset[0]['chosen'][:300])

    # TRL 0.24의 DPOTrainer.__init__(dpo_trainer.py:405)가
    # `model.warnings_issued["estimate_tokens"] = True`를 실행하는데, 이 속성은
    # transformers 5.x에서 삭제됐다. PEFT 래퍼는 __getattr__을 베이스 모델로 넘기므로
    # 양쪽 모두에 빈 dict를 심어 두면 조회가 성공한다. 학습 자체와는 무관한 로깅용 속성.
    for m in (model, getattr(model, 'base_model', None),
              getattr(getattr(model, 'base_model', None), 'model', None)):
        if m is not None and not hasattr(m, 'warnings_issued'):
            m.warnings_issued = {}

    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # PEFT 어댑터를 끈 상태가 곧 참조 모델 (메모리 절약)
        processing_class=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            output_dir=CONFIG['output_dir'],
            run_name=RUN_NAME,
            report_to='wandb',
            beta=CONFIG['beta'],
            max_length=CONFIG['max_seq_len'],
            max_prompt_length=CONFIG['max_prompt_len'],
            per_device_train_batch_size=CONFIG['per_device_batch'],
            gradient_accumulation_steps=CONFIG['grad_accum'],
            num_train_epochs=CONFIG['epochs'],
            learning_rate=CONFIG['learning_rate'],
            lr_scheduler_type='cosine',
            warmup_ratio=0.05,
            logging_steps=10,
            save_steps=200,          # 러너 중단 대비 주기 저장
            save_total_limit=2,
            seed=CONFIG['seed'],
            bf16=True,
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
    print('다음: uv run python remote/eval_vllm.py --mode both --adapter ' + final_dir)


if __name__ == '__main__':
    main()
