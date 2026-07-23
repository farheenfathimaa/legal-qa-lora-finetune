import os
import argparse
import sys
import yaml
import torch
import mlflow
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data_loader import load_and_preprocess_data

def train(config_path="../configs/lora_config.yaml"):
    # Fix paths so script can be run from anywhere
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_config_path = os.path.join(base_dir, config_path.lstrip("../"))
    
    with open(full_config_path, "r") as f:
        config = yaml.safe_load(f)

    # MLflow Setup
    mlflow.set_experiment(config["run_name"])
    mlflow.transformers.autolog()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Prepare Data
    dataset_full_path = os.path.join(base_dir, config["dataset_path"])
    train_dataset, val_dataset = load_and_preprocess_data(
        dataset_full_path, 
        tokenizer, 
        max_length=config["max_seq_length"]
    )
    
    # Check for bitsandbytes and GPU
    try:
        from transformers import BitsAndBytesConfig
        has_bnb = True
    except ImportError:
        has_bnb = False
        
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    # BitsAndBytes 4-bit Quantization Config (if available)
    if has_bnb and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            quantization_config=bnb_config,
            device_map=device_map,
        )
        # Prepare model for PEFT
        model = prepare_model_for_kbit_training(model)
    else:
        print("WARNING: BitsAndBytes or CUDA not found. Loading model in standard fp32 mode for CPU testing.")
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            device_map=device_map,
        )
    
    # LoRA Config
    # Filter target modules depending on model type (e.g. GPT2 uses c_attn instead of q_proj)
    target_modules = config["lora"]["target_modules"]
    if "gpt2" in config["model_name"].lower():
        target_modules = ["c_attn"]
        
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=target_modules,
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Output directory
    output_dir = os.path.join(base_dir, config["output_dir"])

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        optim="adamw_torch" if not has_bnb else config["training"]["optim"],
        evaluation_strategy="steps",
        eval_steps=config["training"]["logging_steps"],
        report_to="mlflow",
        # fp16 might not work well on CPU
        fp16=torch.cuda.is_available(),
        run_name=config["run_name"],
        max_steps=config["training"].get("max_steps", -1), # Allows manual limit for tests
    )

    # Data Collator for Causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    with mlflow.start_run():
        mlflow.log_params(config["training"])
        mlflow.log_params(config["lora"])
        
        # Train
        trainer.train()
        
        # Save model
        final_output_path = os.path.join(output_dir, "final_adapter")
        model.save_pretrained(final_output_path)
        tokenizer.save_pretrained(final_output_path)
        mlflow.log_artifact(final_output_path)
        print(f"Model saved to {final_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml")
    args = parser.parse_args()
    train(args.config)
