"""GRPO 강화학습 파일럿 (H17) — SFT 3연속 실패 후 근본 전환 트랙 A.

핵심: KL 제약으로 베이스 정책 근처에 묶어두고(기존 조율 보존),
정답 도달 여부만 보상해 다양성을 해치지 않으면서 정확도를 올린다.

사용 (서버, tmux):
    nohup uv run python remote/train_grpo.py > train_grpo.log 2>&1 &
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from answer_extract import BS, extract_answer

CONFIG = dict(
    base_model='Qwen/Qwen2.5-3B-Instruct',
    output_dir='outputs/grpo_scale',
    n_problems=6000,        # 스케일업: 파일럿(3000)의 2배 범위
    max_prompt_len=768,
    max_completion_len=1024,
    num_generations=4,      # 문제당 생성 수 (그룹 상대 보상)
    lr=5e-6,                # RL은 SFT보다 훨씬 낮게
    max_steps=1500,
    per_device_batch=4,     # = num_generations 배수여야 함
    seed=42,
)

SYSTEM_PROMPT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. '
    'The final answer is always an integer. '
    'Put your final integer answer inside ' + BS + 'boxed{}.'
)


def build_dataset():
    """중간 난이도 문제 선별: RFT(6회 시도)에서 1~2개만 정답이었던 문제 우선.

    이유: 전부 성공(쉬움)·전부 실패(보상 全0 → 그래디언트 없음) 문제는 GRPO 학습 신호가 약함.
    """
    import json as _json
    df = pd.read_csv('deep-learning-challenge-2026/deep_chal_math_train.csv')
    df.columns = df.columns.str.strip()
    val_ids = set(df.sample(500, random_state=123)['id'])
    bad = 'deep-learning-challenge-2026/train_filtered_ids.csv'
    bad_ids = set(pd.read_csv(bad)['id']) if os.path.exists(bad) else set()

    kept_count = {}
    if os.path.exists('data/sft.jsonl'):
        for line in open('data/sft.jsonl', encoding='utf-8'):
            pid = _json.loads(line)['id']
            kept_count[pid] = kept_count.get(pid, 0) + 1

    df = df[~df['id'].isin(val_ids | bad_ids)].reset_index(drop=True)
    df['kept'] = df['id'].map(kept_count).fillna(0)
    mid = df[df['kept'].isin([1, 2])]                      # 중간 난이도
    rest = df[~df.index.isin(mid.index)]
    take = pd.concat([mid, rest.sample(frac=1, random_state=42)]).head(CONFIG['n_problems'])
    print(f'GRPO 대상 {len(take)}문제 (중간 난이도 {min(len(mid), len(take))}개 포함)')
    return take.reset_index(drop=True)


def main():
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    problems = build_dataset()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG['base_model'],
        max_seq_length=CONFIG['max_prompt_len'] + CONFIG['max_completion_len'],
        load_in_4bit=True,
        fast_inference=True,   # unsloth 내장 vLLM 생성 (GRPO 필수적 속도)
        max_lora_rank=16,
        gpu_memory_utilization=0.6,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        random_state=CONFIG['seed'],
    )

    answers = {r.id: int(r.answer) for r in problems.itertuples()}

    def make_prompt(row):
        return tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user', 'content': row['question']}],
            tokenize=False, add_generation_prompt=True)

    ds = Dataset.from_dict({
        'prompt': [make_prompt(r) for _, r in problems.iterrows()],
        'pid': list(problems['id']),
    })

    def reward_correct(completions, pid, **kwargs):
        """정답 도달 +1, 그 외 0. (형식 보너스: boxed 존재 시 +0.1)"""
        rewards = []
        for text, p in zip(completions, pid):
            ans = extract_answer(text)
            r = 0.0
            if 'boxed' in text:
                r += 0.1
            if ans is not None and ans == answers.get(p):
                r += 1.0
            rewards.append(r)
        return rewards

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_correct],
        train_dataset=ds,
        args=GRPOConfig(
            output_dir=CONFIG['output_dir'],
            run_name='grpo_scale_lr5e-6_s1500',
            report_to='wandb',
            learning_rate=CONFIG['lr'],
            per_device_train_batch_size=CONFIG['per_device_batch'],
            num_generations=CONFIG['num_generations'],
            max_prompt_length=CONFIG['max_prompt_len'],
            max_completion_length=CONFIG['max_completion_len'],
            max_steps=CONFIG['max_steps'],
            logging_steps=5,
            save_steps=100,
            seed=CONFIG['seed'],
            bf16=True,
        ),
    )
    trainer.train()

    final = os.path.join(CONFIG['output_dir'], 'grpo_scale_final')
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print('저장 완료:', final)


if __name__ == '__main__':
    main()
