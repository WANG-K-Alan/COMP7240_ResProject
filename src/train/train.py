# -*- coding: UTF-8 -*-
'''
@Project ：7240_ResProject 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：Winston·H DONG
@Date    ：2026/4/18 21:35 
'''
# -*- coding: UTF-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from torch.utils.data import Dataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, TaskType

# ---- 路径设置（按你之前项目结构） ----
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = os.path.join(ROOT_DIR, "checkpoints", "bert_sentiment_lora")

# ---- 1. 读取 csv 数据：id, text, label ----
csv_path = os.path.join(DATA_DIR, "train.csv")
df = pd.read_csv(csv_path)

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()   # 0/1

print(f"Loaded {len(texts)} samples from train.csv")

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ---- 2. tokenizer ----
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(X_train, y_train, tokenizer)
val_dataset   = TextDataset(X_val,   y_val,   tokenizer)

# ---- 3. 基础 BERT 分类模型 + LoRA 配置 ----
base_model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # 序列分类任务
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]  # 常见设置：只对注意力里的 q,v 加 LoRA
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # 打印有多少参数在训练

# ---- 4. 指标计算函数 ----
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# ---- 5. 训练参数 ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=32,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_steps=50,
    fp16=True,
)

# ---- 6. Trainer ----
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ---- 7. 开始训练 ----
trainer.train()

# ---- 8. 保存 LoRA 微调后的模型和 tokenizer ----
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA sentiment model saved to {OUTPUT_DIR}")

