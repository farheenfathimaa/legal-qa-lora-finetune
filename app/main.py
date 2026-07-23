import yaml
import torch
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="Legal QA API", description="API serving fine-tuned LoRA model for Legal QA")

# Global variables for model and tokenizer
model = None
tokenizer = None

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@app.on_event("startup")
def load_model():
    global model, tokenizer
    try:
        # FastAPI starts typically in the root of the project
        config_path = "configs/lora_config.yaml"
        if not os.path.exists(config_path):
            config_path = "../configs/lora_config.yaml"
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes
            has_bnb = True
        except ImportError:
            has_bnb = False
            
        print("Loading base model...")
        if has_bnb and torch.cuda.is_available():
            base_model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                load_in_4bit=True,
                device_map="auto"
            )
        else:
            print("WARNING: GPU/bitsandbytes not found. Loading in CPU standard mode.")
            base_model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                device_map="cpu"
            )
        
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Try to resolve adapter path from root or from app directory
        adapter_path = config["output_dir"] + "/final_adapter"
        if not os.path.exists(adapter_path):
            adapter_path = "../" + config["output_dir"] + "/final_adapter"
        
        try:
            print("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load adapter. Using base model instead. Error: {e}")
            model = base_model
            
    except Exception as e:
        print(f"Error during model initialization: {e}")

@app.get("/health")
def health_check():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")
        
    prompt = f"### Instruction:\n{request.question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)
        
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_response[len(prompt):].strip()
    
    return AskResponse(answer=answer)
