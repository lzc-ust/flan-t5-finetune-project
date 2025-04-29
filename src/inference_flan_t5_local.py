# src/inference_flan_t5_local.py

import os
import json
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def load_dataset_from_jsonl(file_path):
    """Load input data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_predictions(predictions, output_path):
    """Save predictions to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    # Paths
    model_dir = os.path.join("checkpoints", "flan-t5-debug-local")
    input_data_path = os.path.join("datasets", "processed_dataset.jsonl")
    output_predictions_path = os.path.join("datasets", "predicted_answers_local.jsonl")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    # Create inference pipeline
    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if model.device.type == "cuda" else -1,  # 自动识别是否用GPU
    )

    # Load input data
    dataset = load_dataset_from_jsonl(input_data_path)
    
    # Optional: 只选前30条测试，加快推理速度
    dataset = dataset[:30]

    predictions = []
    for sample in tqdm(dataset, desc="Running inference"):
        input_text = sample["input"]
        generated = generator(input_text, max_new_tokens=128)[0]["generated_text"]

        predictions.append({
            "input": input_text,
            "predicted_answer": generated,
            "reference_answer": sample["output"],
        })

    # Save predictions
    save_predictions(predictions, output_predictions_path)

    print(f"✅ Inference complete! Predictions saved to {output_predictions_path}")

if __name__ == "__main__":
    main()
