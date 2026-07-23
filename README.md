# Legal QA LoRA Fine-tuning

This project implements a complete end-to-end pipeline for fine-tuning a Large Language Model (Mistral-7B) on a Legal Question and Answer dataset using Low-Rank Adaptation (LoRA). Fine-tuning foundational models heavily taxes compute resources; LoRA provides a parameter-efficient approach that freezes the base model weights while only updating a small number of added parameters. This allows for viable training on consumer hardware while matching full fine-tuning performance.

## Architecture

```text
[ Synthetic Legal QA JSONL ]
         │ (data loader)
         ▼
[ HF Datasets Tokenization ] ───┐
                                │
[ Mistral-7B-v0.1 Base ] ───────┼──> [ PEFT LoRA Fine-Tuning ]
(4-bit Quantization)            │             │
                                │             ▼
[ configs/lora_config.yaml ] ───┘    [ MLflow Tracking Server ]
                                              │
                                              ▼
                             [ FastAPI Model Serving Endpoint ]
```

## LoRA Configuration

The model is trained using the following hyperparameters, defined in `configs/lora_config.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Quantization** | 4-bit (NF4) | BitsAndBytes double quantization with fp16 compute |
| **LoRA Rank (r)** | 8 | Rank of the update matrices |
| **LoRA Alpha** | 16 | Scaling factor for the LoRA adapter |
| **Target Modules** | `q_proj`, `v_proj` | Attention components to attach adapters |
| **Learning Rate** | `2.0e-4` | Base learning rate for AdamW |
| **Epochs** | 1 | Number of training passes through the dataset |

## Evaluation
An automated script evaluates both the base model and fine-tuned model variants on a held-out test split, utilizing the ROUGE-L metric.

| Model | ROUGE-L Score |
|-------|---------------|
| Mistral-7B Base | ~0.XXXX |
| LoRA Fine-Tuned | ~0.XXXX |
*(See `eval_results.md` for real metrics after a full training run)*

## Setup Instructions

1. **Clone Repo**
   ```bash
   git clone <your-repo-url>
   cd legal-qa-lora-finetune
   ```

2. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Data Generation** (If you don't already have the dataset)
   ```bash
   cd data
   python generate_dataset.py
   cd ..
   ```

## Usage: Training Pipeline

1. **Run Fine-tuning script:**
   The training script loads the YAML configuration, builds the adapters and tracks all metrics to MLflow.
   ```bash
   cd src
   python train.py
   ```

2. **Monitor with MLflow:**
   View hyperparameter tracking, loss curves, and artifact logging.
   ```bash
   mlflow ui
   ```
   Open `http://localhost:5000` in your browser. Look for the `legal-qa-lora-finetune` experiment to view loss progressions and adapter saves.

3. **Evaluate the Model:**
   ```bash
   cd src
   python eval.py
   ```
   This generates `eval_results.md` with base vs fine-tuned comparative metrics.

## Usage: Serving (FastAPI)

1. **Start the API locally:**
   ```bash
   cd app
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Testing via cURL:**
   ```bash
   curl -X POST "http://localhost:8000/ask" \
   -H "Content-Type: application/json" \
   -d '{"question":"What is a statute of limitations?"}'
   ```

## Known Limitations
* **Dataset Size:** The included data generation script provides a synthetic set of only 250 samples, which is sufficient for demonstrating the pipeline but heavily restrictive for production robustness.
* **Model Choices & VRAM:** Mistral-7B at 4-bit uses roughly 6-7 GB VRAM at inference and up to 12 GB during training. Depending on hardware limitations, sequence lengths or batch sizes may need further reduction.
