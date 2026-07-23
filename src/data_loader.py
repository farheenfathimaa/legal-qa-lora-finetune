from datasets import load_dataset

def format_instruction_prompt(example):
    """
    Format the dataset into an instruction based prompt.
    """
    if example.get("input", "") != "":
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    
    return {"text": text}

def load_and_preprocess_data(data_path, tokenizer, max_length=512):
    """
    Loads JSONL dataset from the given path, formats it, tokenizes, and returns HF dataset.
    """
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # Format the prompts
    dataset = dataset.map(format_instruction_prompt)
    
    # Validation split for evaluation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    def tokenize_function(examples):
        # We add padding and truncation
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    
    tokenized_train_dataset = dataset["train"].map(tokenize_function, batched=True)
    tokenized_val_dataset = dataset["test"].map(tokenize_function, batched=True)
    
    return tokenized_train_dataset, tokenized_val_dataset

if __name__ == "__main__":
    from transformers import AutoTokenizer
    # Basic sanity check
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    # Quick fix for tokenizer padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    train_ds, val_ds = load_and_preprocess_data("../data/dataset.jsonl", tokenizer)
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Sample train text:\n{train_ds[0]['text']}")
