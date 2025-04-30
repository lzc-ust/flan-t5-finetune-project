from evaluate import load
import numpy as np
from transformers import AutoTokenizer

# 初始化 tokenizer（确保与训练时保持一致）
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# 加载评估器
rouge = load("rouge")
bleu = load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # 解码预测结果
    decoded_preds = [pred.strip() for pred in tokenizer.batch_decode(predictions, skip_special_tokens=True)]

    # 将 -100 的 label 替换为 pad_token_id，再解码
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = [label.strip() for label in tokenizer.batch_decode(labels, skip_special_tokens=True)]

    # 计算 ROUGE 和 BLEU 分数
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bleu_result = bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])

    result = {
        "rougeL": round(rouge_result["rougeL"] * 100, 2),
        "bleu": round(bleu_result["bleu"] * 100, 2),
    }
    return result