# tests/evaluate_on_testset.py

import os
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import evaluate
from tqdm import tqdm

def load_dataset_from_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

def find_latest_checkpoint(base_dir):
    checkpoints = [d for d in os.listdir(base_dir) if d.startswith("flan-t5-full")]
    checkpoints.sort(reverse=True)
    return os.path.join(base_dir, checkpoints[0]) if checkpoints else None

def main():
    # 注意：路径要从 tests/ 回到上级目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_path = os.path.join(project_root, "datasets", "test.jsonl")
    model_base = os.path.join(project_root, "checkpoints")
    model_dir = find_latest_checkpoint(model_base)
    output_dir = os.path.join(project_root, "tests", "test_results")
    os.makedirs(output_dir, exist_ok=True)

    if model_dir is None:
        raise ValueError("❌ No model checkpoint found in 'checkpoints/'")

    print(f"✅ Using latest model: {model_dir}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()

    # Load test dataset
    test_dataset = load_dataset_from_jsonl(test_path)

    # Create pipeline for generation
    gen_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if model.device.type == "cuda" else -1,
        max_new_tokens=128
    )

    # Generate predictions
    preds = []
    refs = []
    inputs = []
    for sample in tqdm(test_dataset, desc="Predicting"):
        input_text = sample["input"]
        output_text = sample["output"]
        result = gen_pipe(input_text)[0]["generated_text"].strip()
        preds.append(result)
        refs.append(output_text.strip())
        inputs.append(input_text.strip())

    # Save predictions to CSV
    df = pd.DataFrame({
        "input": inputs,
        "prediction": preds,
        "reference": refs
    })
    csv_path = os.path.join(output_dir, "test_predictions.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Evaluate metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    rouge_result = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
    bleu_result = bleu.compute(predictions=preds, references=[[ref] for ref in refs])
    meteor_result = meteor.compute(predictions=preds, references=refs)

    simple_acc = sum([ref.lower() in pred.lower() or pred.lower() in ref.lower() for pred, ref in zip(preds, refs)]) / len(preds)

    # Display results
    print("\n✅ Test Set Evaluation Complete!\n")
    print(f"Simple Containment Accuracy: {simple_acc * 100:.2f}%")
    print(f"ROUGE-L F1 Score: {rouge_result['rougeL'] * 100:.2f}%")
    print(f"BLEU Score: {bleu_result['bleu'] * 100:.2f}%")
    print(f"METEOR Score: {meteor_result['meteor'] * 100:.2f}%")
    print(f"📄 Predictions saved to: {csv_path}")

    # Save metrics to JSON
    json_path = os.path.join(output_dir, "test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "rougeL": round(rouge_result['rougeL'] * 100, 2),
            "bleu": round(bleu_result['bleu'] * 100, 2),
            "meteor": round(meteor_result['meteor'] * 100, 2),
            "simple_accuracy": round(simple_acc * 100, 2)
        }, f, indent=4, ensure_ascii=False)
    print(f"📊 Metrics saved to: {json_path}")

if __name__ == "__main__":
    main()
