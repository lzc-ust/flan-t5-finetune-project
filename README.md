# Flan-T5 Fine-tuning for Software Engineering QA

This project fine-tunes [Flan-T5](https://huggingface.co/google/flan-t5-large) on a curated dataset of software engineering interview questions. It supports local training, evaluation, and batch inference, and was further scaled using the HKUST SuperPOD GPU cluster.

---

## 🗂️ Project Structure

```
├── checkpoints/                 # Saved model checkpoints (auto-generated)
├── datasets/                   # Processed training/validation/test data and predictions
│   ├── train.json              # Training data (80%)
│   ├── valid.json              # Validation data (10%)
│   ├── test.json               # Test data (10%)
│   ├── predicted_answers_local.jsonl  # Model outputs on test set
│   ├── predicted_answers_local.csv
│   └── Software Questions.csv  # Original dataset before formatting
├── logs/                       # Slurm output/error logs from SuperPOD
├── scripts/
│   └── train_flan_t5.slurm     # SLURM script for training on SuperPOD
├── src/                        # Core source code
│   ├── train_flan_t5.py        # Training script (for general use)
│   ├── train_flan_t5_local.py  # Training script (local GPU environment)
│   ├── preprocess_dataset.py   # Data formatting and splitting
│   ├── inference_flan_t5_local.py    # Run inference on test set
│   ├── inference_to_csv_local.py     # Convert predictions to CSV
│   └── compute_metrics.py      # Compute BLEU, ROUGE, METEOR
├── tests/
│   └── run_eval_testset.sh     # Evaluation pipeline
├── init_env.sh                # Conda environment initialization
├── requirements.txt           # Main dependencies
├── requirements-freeze.txt    # Frozen environment (pip freeze)
└── README.md                  # You're here :)
```

---

## 🚀 Setup Instructions

### 1. Create and activate environment

```bash
bash init_env.sh
conda activate flan-t5-finetune
```

Or manually:

```bash
conda create -n flan-t5-finetune python=3.10
pip install -r requirements.txt
```

---

## 📦 Data Preparation

To preprocess and split the dataset:

```bash
python src/preprocess_dataset.py
```

This will generate:

- `train.json`, `valid.json`, `test.json` under `datasets/`
- Additional `.jsonl` and `.csv` files for inference output

---

## 🏋️ Model Training

### ✅ Local GPU

```bash
python src/train_flan_t5_local.py
```

### ✅ SuperPOD (via SLURM)

```bash
sbatch scripts/train_flan_t5.slurm
```

---

## 🔍 Evaluation Pipeline

Run evaluation on test set:

```bash
python tests/evaluate_on_testset.py
```

This will:

- Run inference using best model
- Save results to `datasets/predicted_answers_local.jsonl`
- Compute metrics and save to `tests/test_results/`

---

## 📊 Metrics

Evaluation includes:

- BLEU
- ROUGE-L
- METEOR
- Exact match accuracy (optional)

Computed using `src/compute_metrics.py`.

---

## 🧠 Prompt Format Example

Each input is reformulated as:

```text
Answer the following software engineering interview question:
You are answering a Medium General Programming question.
Question: What is polymorphism?
```

---

## ✅ Future Work

- [ ] Extend model to `flan-t5-xl`
- [ ] Collect more questions from diverse domains (DevOps, Testing, Frontend)
- [ ] Build interactive web demo for real-time QA
- [ ] Explore RLHF-based finetuning

---

## 📬 Contact

- Ziling Wang: zwanglm@connect.ust.hk  
- Zhicheng Li: zlicv@connect.ust.hk
