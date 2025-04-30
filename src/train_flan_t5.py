import os
import json
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from src.compute_metrics import compute_metrics

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
    labels = tokenizer(
        examples["output"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    # Timestamp for unique output/log directories
    time_tag = datetime.now().strftime("%Y%m%d-%H%M")

    # Paths
    train_path = os.path.join("datasets", "train.jsonl")
    valid_path = os.path.join("datasets", "valid.jsonl")
    output_dir = os.path.join("checkpoints", f"flan-t5-full-{time_tag}")
    logs_dir = os.path.join("logs", f"log-{time_tag}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Load dataset
    train_dataset = load_dataset_from_jsonl(train_path)
    valid_dataset = load_dataset_from_jsonl(valid_path)

    # Load model and tokenizer
    model_checkpoint = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # Preprocess
    tokenized_train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_valid_dataset = valid_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_dir=logs_dir,
        logging_steps=50,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        report_to="tensorboard",
        save_safetensors=True,
        # gradient_checkpointing=True,  # Uncomment if needed to save GPU memory
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Start Training
    trainer.train()

    # Save final model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Training complete. Best model saved at {output_dir}")

if __name__ == "__main__":
    main()