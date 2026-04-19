# -*- coding: UTF-8 -*-
"""
@Project ：7240_ResProject
@File    ：recommend_pos_lora.py
@IDE     ：PyCharm
@Author  ：Winston·H DONG
@Date    ：2026/4/19 19:18
"""
import os
import torch
import torch.nn.functional as F
import pandas as pd

from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel

# Project root and data paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Directory where the LoRA fine-tuned model is saved
LORA_MODEL_DIR = os.path.join(ROOT_DIR, "src/model", "bert_sentiment_lora")

# Path to store precomputed positive-review embeddings
CORPUS_PATH = os.path.join(DATA_DIR, "positive_corpus_embeddings_lora.pt")


def load_lora_model():
    """
    Load the base BERT classification model and apply LoRA weights,
    then load the tokenizer.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load base BERT classification model
    base_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
    )
    # 2) Load LoRA adapter weights
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
    model.to(device)
    model.eval()

    # 3) Load tokenizer from the same directory (fallback to base tokenizer)
    if os.path.exists(LORA_MODEL_DIR):
        tokenizer = BertTokenizer.from_pretrained(LORA_MODEL_DIR)
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer, device


def encode_texts_with_lora(
    model,
    tokenizer,
    texts,
    device,
    max_len=256,
    batch_size=32,
):
    """
    Encode multiple texts using the BERT encoder part of the LoRA model.
    Returns a tensor of shape [N, hidden].

    Note: we only use the BERT encoder and take [CLS] embeddings,
    ignoring the classification head.
    """
    model.eval()
    all_vecs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Use only the BERT encoder (ignore classifier head)
            outputs = model.base_model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            cls = outputs.last_hidden_state[:, 0, :]  # [B, hidden]
            all_vecs.append(cls.cpu())

    return torch.cat(all_vecs, dim=0)  # [N, hidden]


def build_positive_corpus_embeddings_lora():
    """
    1. Read data/train.csv
    2. Filter rows where label == 1 (positive reviews)
    3. Encode texts with the LoRA BERT encoder to obtain embeddings
    4. Save them to positive_corpus_embeddings_lora.pt
    """
    model, tokenizer, device = load_lora_model()

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    pos_df = df[df["label"] == 1].reset_index(drop=True)

    if len(pos_df) == 0:
        raise ValueError("No samples with label == 1 found in train.csv.")

    texts = pos_df["text"].astype(str).tolist()
    ids = pos_df["id"].tolist()

    print(f"Number of positive reviews: {len(texts)}. Start encoding (LoRA model)...")

    embeddings = encode_texts_with_lora(
        model, tokenizer, texts, device,
        max_len=256, batch_size=32
    )

    torch.save(
        {
            "embeddings": embeddings,  # [N, hidden]
            "ids": ids,
            "texts": texts,
        },
        CORPUS_PATH,
    )
    print(f"Positive-review embeddings saved to: {CORPUS_PATH}")


def recommend_from_positive_lora(query_text, top_k=5):
    """
    Given a new review text, encode it with the LoRA model and
    find the top_k most similar reviews in the precomputed positive-review corpus.

    Returns:
        List of dicts: [{'id':..., 'text':..., 'score':...}, ...]
    """
    model, tokenizer, device = load_lora_model()

    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(
            f"{CORPUS_PATH} not found. Please run build_positive_corpus_embeddings_lora() first."
        )

    data = torch.load(CORPUS_PATH)
    corpus_embeddings = data["embeddings"]  # [N, hidden] (stored on CPU)
    ids = data["ids"]
    texts = data["texts"]

    # Encode query text
    enc = tokenizer(
        [query_text],
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.base_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        q_vec = outputs.last_hidden_state[:, 0, :]  # [1, hidden]

    # Cosine similarity
    q_vec = F.normalize(q_vec, p=2, dim=-1)             # [1, hidden]
    c_vec = F.normalize(corpus_embeddings, p=2, dim=-1) # [N, hidden]
    sims = torch.mm(q_vec.cpu(), c_vec.T).squeeze(0)    # [N]

    top_k = min(top_k, len(sims))
    topk_vals, topk_idx = torch.topk(sims, k=top_k)

    results = []
    for score, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
        results.append(
            {
                "id": ids[idx],
                "text": texts[idx],
                "score": float(score),
            }
        )
    return results


if __name__ == "__main__":
    # First time: build the positive-review embedding corpus
    build_positive_corpus_embeddings_lora()

    # Afterwards: call the recommendation function multiple times
    q = "This movie is really heartwarming and beautiful."
    recs = recommend_from_positive_lora(q, top_k=5)
    for r in recs:
        print("id:", r["id"], "score:", r["score"])
        print(r["text"][:200].replace("\n", " "), "...")
        print("=" * 60)
