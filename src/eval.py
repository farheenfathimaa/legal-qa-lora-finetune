import json
import yaml
import os
import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_loader import load_and_preprocess_data

def evaluate_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "configs/lora_config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Get Test Data
    dataset_full_path = os.path.join(base_dir, config["dataset_path"])
    _, val_dataset = load_and_preprocess_data(
        dataset_full_path, 
        tokenizer, 
        max_length=config["max_seq_length"]
    )
    
    # We will pick 5 samples for manual review in the markdown file
    eval_samples = val_dataset.select(range(min(5, len(val_dataset))))
    
    # 3. Load Metrics
    rouge = evaluate.load("rouge")

    # 4. Load Base Model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        load_in_4bit=True,
        device_map="auto"
    )

    def generate_answers(model, samples):
        results = []
        for sample in samples:
            text = sample['text']
            # We want to prompt the model but need to extract just the instruction to avoid feeding the actual label
            # The prompt formatting in data_loader creates: "### Instruction:... ### Response:..."
            prompt, reference = text.split("### Response:\n")
            prompt += "### Response:\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
            
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extracted predicted response
            pred = full_response[len(prompt):].strip()
            
            results.append({
                "prompt": prompt,
                "reference": reference.strip(),
                "predicted": pred
            })
        return results

    print("Generating base model answers...")
    base_results = generate_answers(base_model, eval_samples)
    base_preds = [r["predicted"] for r in base_results]
    refs = [r["reference"] for r in base_results]
    
    base_rouge = rouge.compute(predictions=base_preds, references=refs)

    # 5. Load Fine-Tuned Model
    adapter_path = "../" + config["output_dir"] + "/final_adapter"
    try:
        print("Loading fine-tuned model...")
        ft_model = PeftModel.from_pretrained(base_model, adapter_path)
        print("Generating fine-tuned model answers...")
        ft_results = generate_answers(ft_model, eval_samples)
        ft_preds = [r["predicted"] for r in ft_results]
        ft_rouge = rouge.compute(predictions=ft_preds, references=refs)
    except Exception as e:
        print(f"Fine-tuned model could not be loaded (likely not trained yet during sanity check). Error: {e}")
        ft_results = [{"predicted": "N/A"} for _ in eval_samples]
        ft_rouge = {"rougeL": 0.0}

    # 6. Generate Markdown Output
    md_content = "# Evaluation Results\n\n"
    md_content += "## ROUGE-L Scores\n\n"
    md_content += "| Model | ROUGE-L |\n"
    md_content += "|---|---|\n"
    md_content += f"| Base Model ({config['model_name']}) | {base_rouge.get('rougeL', 0):.4f} |\n"
    md_content += f"| Fine-Tuned (LoRA) | {ft_rouge.get('rougeL', 0):.4f} |\n\n"
    
    md_content += "## Qualitative Comparison (Sample Results)\n\n"
    for i, res in enumerate(base_results):
        prompt_snippet = res["prompt"].replace("\n", " ")[:100] + "..."
        md_content += f"### Sample {i+1}: {prompt_snippet}\n"
        md_content += f"- **Reference:** {res['reference']}\n"
        md_content += f"- **Base Model:** {res['predicted']}\n"
        md_content += f"- **Fine-Tuned:** {ft_results[i]['predicted']}\n\n"

    with open("../eval_results.md", "w") as f:
        f.write(md_content)
        
    print("Evaluation finished. Results saved to eval_results.md")

if __name__ == "__main__":
    evaluate_models()
