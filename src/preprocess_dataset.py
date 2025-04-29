# preprocess_dataset.py

import pandas as pd
import json
import os

def preprocess_and_save(input_csv_path, output_jsonl_path, long_answer_threshold=50):
    # Step 1: 读取CSV
    df = pd.read_csv(input_csv_path, encoding='gbk')
    
    # Step 2: 删除无用字段
    if "Question Number" in df.columns:
        df = df.drop(columns=["Question Number"])
    
    # Step 3: 处理每一行
    data = []
    for _, row in df.iterrows():
        question = row["Question"]
        answer = row["Answer"]
        category = row["Category"]
        difficulty = row["Difficulty"]
        
        # 构建 input prompt
        input_text = f"Answer the following software engineering interview question:\n"
        input_text += f"You are answering a {difficulty} {category} question.\n"
        input_text += f"Question: {question}"
        
        # 如果答案很长，提示需要详细回答
        if len(answer.split()) >= long_answer_threshold:
            input_text += "\nPlease provide a detailed answer."
        
        # 构建一条样本
        sample = {
            "input": input_text,
            "output": answer
        }
        data.append(sample)
    
    # Step 4: 保存为JSONL
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ Finished! Processed {len(data)} samples and saved to {output_jsonl_path}")

# 调用
if __name__ == "__main__":
    input_csv_path = "../datasets/Software Questions.csv"      # 你的原始数据
    output_jsonl_path = "../datasets/processed_dataset.jsonl"   # 输出文件
    preprocess_and_save(input_csv_path, output_jsonl_path)
