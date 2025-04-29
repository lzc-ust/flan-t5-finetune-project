# src/evaluate_predictions_local.py

import os
import json
import evaluate
from tqdm import tqdm

def load_predictions(file_path):
    preds = []
    refs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            preds.append(sample["predicted_answer"].strip())
            refs.append(sample["reference_answer"].strip())
    return preds, refs

def compute_simple_accuracy(preds, refs):
    match_count = 0
    for pred, ref in zip(preds, refs):
        if ref.lower() in pred.lower() or pred.lower() in ref.lower():
            match_count += 1
    return match_count / len(preds)

def compute_rouge(preds, refs):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
    return results["rougeL"]

def compute_bleu(preds, refs):
    bleu = evaluate.load("bleu")
    # BLEU指标需要每个reference是list of tokens
    results = bleu.compute(predictions=preds, references=[[ref] for ref in refs])
    return results["bleu"]

def compute_meteor(preds, refs):
    meteor = evaluate.load("meteor")
    results = meteor.compute(predictions=preds, references=refs)
    return results["meteor"]

def main():
    prediction_path = os.path.join("datasets", "predicted_answers_local.jsonl")

    preds, refs = load_predictions(prediction_path)

    simple_acc = compute_simple_accuracy(preds, refs)
    rouge_l = compute_rouge(preds, refs)
    bleu_score = compute_bleu(preds, refs)
    meteor_score = compute_meteor(preds, refs)

    print("✅ Evaluation complete!\n")
    print(f"Simple Containment Accuracy: {simple_acc * 100:.2f}%")
    print(f"ROUGE-L F1 Score: {rouge_l * 100:.2f}%")
    print(f"BLEU Score: {bleu_score * 100:.2f}%")
    print(f"METEOR Score: {meteor_score * 100:.2f}%")

if __name__ == "__main__":
    main()
