# Legal QA LoRA Fine-Tuning

A complete, production-ready pipeline for fine-tuning a Large Language Model (Mistral-7B) on a Legal Q&A dataset using LoRA (Low-Rank Adaptation). The project covers everything from data generation, efficient adapter training with 4-bit quantization, MLflow experiment tracking, before/after evaluation, and a FastAPI REST API for serving the fine-tuned model.

## Architecture

```
[ Synthetic Legal QA JSONL ]
         │ (src/data_loader.py)
         ▼
[ HF Datasets Tokenization ]
         │
[ Mistral-7B Base Model ] ──────> [ PEFT LoRA Adapters ]
  (4-bit NF4 quantization)                  │
         │                                  ▼
[ configs/lora_config.yaml ]    [ MLflow Tracking Server ]
  (rank, alpha, lr, epochs)         (params + loss + artifacts)
                                            │
                                            ▼
                               [ FastAPI /ask Endpoint ]
                                (base model + LoRA adapter)
```

## LoRA Configuration

All hyperparameters are defined in [`configs/lora_config.yaml`](configs/lora_config.yaml):

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Base Model** | `mistralai/Mistral-7B-v0.1` | Foundation model |
| **Quantization** | 4-bit NF4 | Via BitsAndBytes (requires CUDA GPU) |
| **LoRA Rank (r)** | 8 | Low-rank update matrix size |
| **LoRA Alpha** | 16 | Scaling factor for adapters |
| **Target Modules** | `q_proj`, `v_proj` | Attention layers for adapter injection |
| **LoRA Dropout** | 0.05 | Regularization |
| **Learning Rate** | `2e-4` | AdamW optimizer |
| **Epochs** | 1 | Training passes through dataset |
| **Optimizer** | `paged_adamw_32bit` | Memory-efficient on GPU |

## Evaluation

After training, `src/eval.py` compares base model vs fine-tuned model on a held-out test split using **ROUGE-L**, a recall-oriented metric that measures longest common subsequences between generated and reference answers.

| Model | ROUGE-L Score |
|-------|---------------|
| Mistral-7B Base | See `eval_results.md` |
| LoRA Fine-Tuned | See `eval_results.md` |

> Run `python src/eval.py` after training to populate `eval_results.md` with real scores.

## Setup

```bash
# Clone the repo
git clone https://github.com/farheenfathimaa/legal-qa-lora-finetune.git
cd legal-qa-lora-finetune

# Create and activate conda environment (recommended)
conda create -n legal_lora python=3.10 -y
conda activate legal_lora

# Install dependencies
pip install -r requirements.txt
```

> **GPU Note:** Mistral-7B with 4-bit quantization requires ~12 GB VRAM for training and ~7 GB for inference. The scripts automatically fall back to CPU/fp32 mode if no GPU is detected (useful for testing with a smaller model like `gpt2` by editing `configs/lora_config.yaml`).

## Usage

### 1. Generate Dataset

```bash
cd data
python generate_dataset.py   # Creates data/dataset.jsonl (250 samples)
cd ..
```

### 2. Train the Model

```bash
python src/train.py
```

All runs are automatically logged to MLflow (loss, hyperparameters, adapter artifacts).

### 3. Monitor with MLflow

```bash
mlflow ui
```

Open **http://localhost:5000** — look for the `legal-qa-lora-finetune` experiment. You'll see per-step training loss curves and the saved adapter artifact.

### 4. Evaluate

```bash
python src/eval.py
```

Outputs a before/after ROUGE-L table saved to `eval_results.md`.

### 5. Serve the API

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```

Test in browser: **http://localhost:8000/docs** (Swagger UI with interactive testing)

Or via PowerShell:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method Post -ContentType "application/json" -Body '{"question": "What is a statute of limitations?"}'
```

Health check:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

## Project Structure

```
legal-qa-lora-finetune/
├── app/
│   └── main.py              # FastAPI serving app
├── configs/
│   └── lora_config.yaml     # All training hyperparameters
├── data/
│   ├── generate_dataset.py  # Synthetic dataset generator
│   └── dataset.jsonl        # 250 legal Q&A instruction samples
├── notebooks/               # For exploratory work
├── src/
│   ├── data_loader.py       # HF dataset loading + tokenization
│   ├── train.py             # LoRA fine-tuning + MLflow logging
│   └── eval.py              # Before/after ROUGE-L evaluation
├── .gitignore
├── requirements.txt
└── README.md
```

## Known Limitations

- **Dataset size:** 250 synthetic samples is a minimal proof-of-concept. Production use would require thousands of expert-curated legal Q&A pairs.
- **Hardware:** Mistral-7B requires a CUDA GPU with at least 12 GB VRAM for training. CPU fallback is provided but only suitable for testing with smaller models.
- **Model choice:** Mistral-7B is a general-purpose base model. A legal-domain pre-trained base (e.g., LegalBERT, Lawyer-LLaMA) would yield stronger results.
