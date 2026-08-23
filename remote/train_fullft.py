"""exp93: Qwen2.5-3B-Instruct full FT (bf16) — NuminaMath 정수답 부분집합.

QLoRA 스크립트(train_qlora.py)를 안 쓰는 이유: exp92의 교훈 — 어댑터는 용량을
어떻게 조절해도 능력 향상과 투표 손상이 분리되지 않았다. 3위 참가자의 레시피는
full FT였고, 이번이 그 재현이다.

Unsloth/TRL의 assistant-only 손실 대신 수동 라벨 마스킹을 쓴다: 렌더링된 텍스트를
마지막 '<|im_start|>assistant' 경계에서 잘라 프롬프트 구간 라벨을 -100으로 채운다.
버전 의존성이 없고 8/15의 손실 마스킹 수정(전체 시퀀스 손실 결함)을 계승한다.

사용 (서버, L40S 48GB):
    nohup uv run python remote/train_fullft.py > exp93_train.log 2>&1 &
OOM이 나면 PER_DEVICE_BATCH를 2로, GRAD_ACCUM을 64로 바꿔 재실행 (유효 배치 유지).
"""
import json
import os

CONFIG = dict(
    base_model='Qwen/Qwen2.5-3B-Instruct',  # 대회 규칙: 고정
    data_path='data/numina_full.jsonl',
    output_dir='outputs/exp93_fullft',
    max_seq_len=2048,
    learning_rate=1e-5,   # 사전등록 고정. 발산 시에만 5e-6 (1회 한정)
    epochs=1,
    per_device_batch=4,
    grad_accum=32,        # 유효 배치 128
    warmup_ratio=0.03,
    seed=42,
    num_checkpoints=4,    # 25/50/75/100% 지점, 모델만 저장
)


def main():
    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(CONFIG['base_model'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['base_model'], torch_dtype=torch.bfloat16,
        attn_implementation='sdpa')
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    marker_ids = tok('<|im_start|>assistant\n', add_special_tokens=False)['input_ids']

    def tokenize(ex):
        text = tok.apply_chat_template(ex['messages'], tokenize=False,
                                       add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False, truncation=True,
                  max_length=CONFIG['max_seq_len'])['input_ids']
        # 마지막 assistant 경계를 찾아 그 앞까지 라벨 마스킹
        cut = -1
        m, L = marker_ids, len(marker_ids)
        for j in range(len(ids) - L, -1, -1):
            if ids[j:j + L] == m:
                cut = j + L
                break
        labels = [-100] * len(ids) if cut < 0 else [-100] * cut + ids[cut:]
        return {'input_ids': ids, 'labels': labels, 'length': len(ids)}

    ds = load_dataset('json', data_files=CONFIG['data_path'], split='train')
    ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=4)
    n_bad = sum(1 for ex in ds.select(range(min(2000, len(ds))))
                if all(l == -100 for l in ex['labels']))
    print(f'학습 샘플 {len(ds)}개 (앞 2000개 중 마스킹 실패 {n_bad}개 — 0이어야 정상)', flush=True)
    assert n_bad == 0, 'assistant 경계 탐지 실패 — 템플릿 확인 필요'

    def collate(batch):
        maxlen = max(len(b['input_ids']) for b in batch)
        pad = tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            k = maxlen - len(b['input_ids'])
            input_ids.append(b['input_ids'] + [pad] * k)
            labels.append(b['labels'] + [-100] * k)
            attn.append([1] * len(b['input_ids']) + [0] * k)
        return {'input_ids': torch.tensor(input_ids),
                'labels': torch.tensor(labels),
                'attention_mask': torch.tensor(attn)}

    steps_per_epoch = len(ds) // (CONFIG['per_device_batch'] * CONFIG['grad_accum'])
    save_steps = max(1, steps_per_epoch // CONFIG['num_checkpoints'])
    print(f'총 스텝 ~{steps_per_epoch}, 체크포인트 간격 {save_steps}', flush=True)

    # transformers v5: group_by_length 제거됨, warmup_ratio는 5.2에서 제거 예정
    args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        per_device_train_batch_size=CONFIG['per_device_batch'],
        gradient_accumulation_steps=CONFIG['grad_accum'],
        num_train_epochs=CONFIG['epochs'],
        learning_rate=CONFIG['learning_rate'],
        lr_scheduler_type='cosine',
        warmup_steps=max(10, int(CONFIG['warmup_ratio'] * steps_per_epoch)),
        bf16=True,
        optim='adamw_bnb_8bit',
        logging_steps=10,
        save_steps=save_steps,
        save_only_model=True,
        seed=CONFIG['seed'],
        report_to=[],
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    trainer.train()
    trainer.save_model(os.path.join(CONFIG['output_dir'], 'final'))
    tok.save_pretrained(os.path.join(CONFIG['output_dir'], 'final'))
    # 체크포인트에서 곧바로 vLLM 로드가 되도록 토크나이저를 각 ckpt에도 복사
    for d in os.listdir(CONFIG['output_dir']):
        p = os.path.join(CONFIG['output_dir'], d)
        if d.startswith('checkpoint-') and os.path.isdir(p):
            tok.save_pretrained(p)
    print('학습 완료', flush=True)


if __name__ == '__main__':
    main()
