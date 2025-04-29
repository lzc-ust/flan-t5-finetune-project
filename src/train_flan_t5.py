from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir="logs/tensorboard",    # ✅ 保存tensorboard日志的目录
    logging_steps=50,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=True,
    dataloader_num_workers=4,
    report_to="tensorboard",            # ✅ 报告到tensorboard
)
