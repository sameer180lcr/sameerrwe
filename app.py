import os
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

@app.on_event("startup")
def load_model():
    global tokenizer, model
    model_name = "google/gemma-3-270m"
    hf_token = os.getenv("HF_TOKEN")  # use env variable

    print("🚀 Loading model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    print("✅ Model loaded!")

@app.get("/health")
def health():
    return {"status": "ok", "message": "API is running"}

@app.get("/ask")
def ask(q: str):
    inputs = tokenizer(q, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"question": q, "answer": answer}

