import evaluate
import numpy as np

# 加载评估指标
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # 解码预测和标签，忽略-100
    decoded_preds = [pred.strip() for pred in predictions]
    decoded_labels = [label.strip() for label in labels]

    # ROUGE评估
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rougeL_f1 = rouge_result["rougeL"].mid.fmeasure

    # BLEU评估
    bleu_result = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    bleu_score = bleu_result["bleu"]

    # 可以扩展其他指标，比如简单准确率 Accuracy（如果你做分类类任务）
    # 这里默认是开放文本生成，不算准确率

    return {
        "rougeL": round(rougeL_f1 * 100, 2),
        "bleu": round(bleu_score * 100, 2),
    }