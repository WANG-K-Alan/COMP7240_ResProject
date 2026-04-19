# -*- coding: UTF-8 -*-
'''
@Project ：7240_ResProject 
@File    ：predict.py
@IDE     ：PyCharm 
@Author  ：Winston·H DONG
@Date    ：2026/4/18 21:39 
'''
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "src/model", "bert_sentiment_lora")

def load_lora_model():
    base_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device

def predict(text: str) -> int:
    model, tokenizer, device = load_lora_model()
    enc = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        pred = torch.argmax(logits, dim=-1).item()
    return int(pred)   # 0 or 1

if __name__ == "__main__":
    print(predict("This movie is absolutely wonderful!"))
