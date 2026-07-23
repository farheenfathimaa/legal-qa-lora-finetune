import os
import yaml
import torch
import mlflow
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data_loader import load_and_preprocess_data

def train():
    # Load config
    with open("../configs/lora_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # MLflow Setup
    mlflow.set_experiment(config["run_name"])
    mlflow.transformers.autolog()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Prepare Data
    train_dataset, val_dataset = load_and_preprocess_data(
        "../" + config["dataset_path"], 
        tokenizer, 
        max_length=config["max_seq_length"]
    )
    
    # BitsAndBytes 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)
    
    # LoRA Config
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="../" + config["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        optim=config["training"]["optim"],
        evaluation_strategy="steps",
        eval_steps=config["training"]["logging_steps"],
        report_to="mlflow",
        fp16=True,
        run_name=config["run_name"]
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
        output_path = "../" + config["output_dir"] + "/final_adapter"
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        mlflow.log_artifact(output_path)
        print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train()
