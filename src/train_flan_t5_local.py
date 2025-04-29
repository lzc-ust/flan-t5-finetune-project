# src/train_flan_t5_local.py

import os
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

def load_dataset_from_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

def preprocess_function(examples, tokenizer, max_input_length=256, max_target_length=64):
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["output"],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    # Paths
    dataset_path = os.path.join("datasets", "processed_dataset.jsonl")
    output_dir = os.path.join("checkpoints", "flan-t5-debug-local")
    logs_dir = "logs"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset_from_jsonl(dataset_path)
    dataset = dataset.select(range(min(len(dataset), 200)))
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    # Load model and tokenizer
    model_checkpoint = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # Preprocess
    tokenized_train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_eval_dataset = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",   # 每个epoch做一次eval
        save_strategy="epoch",         # 每个epoch保存一次checkpoint
        save_total_limit=1,             # 只保留最好的1个checkpoint
        learning_rate=5e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=5,             # 多跑几轮，让模型能看到更多
        weight_decay=0.0,
        logging_dir=logs_dir,
        logging_steps=10,
        predict_with_generate=True,     # 让eval时可以生成预测文本
        fp16=False,                     # 本地不要开混合精度
        push_to_hub=False,
        load_best_model_at_end=True,     # 训练结束后加载最好的那个
        metric_for_best_model="eval_loss",   # 以eval loss最小作为标准
        greater_is_better=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # 如果2次eval没有提升，就停止训练
    )

    # Start Training
    trainer.train()

    # Save final model and tokenizer (again just in case)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Local debug training complete. Best model saved at {output_dir}")

if __name__ == "__main__":
    main()
