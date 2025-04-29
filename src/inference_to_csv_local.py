# src/inference_to_csv_local.py

import os
import json
import pandas as pd

def load_predictions(file_path):
    """Load predicted and reference answers."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            data.append({
                "Input Question": sample.get("input", "").strip(),
                "Reference Answer": sample.get("reference_answer", "").strip(),
                "Predicted Answer": sample.get("predicted_answer", "").strip()
            })
    return data

def main():
    # Paths
    input_jsonl_path = os.path.join("datasets", "predicted_answers_local.jsonl")
    output_csv_path = os.path.join("datasets", "predicted_answers_local.csv")

    # Load data
    data = load_predictions(input_jsonl_path)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save as CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ Inference results saved to {output_csv_path}")

if __name__ == "__main__":
    main()
