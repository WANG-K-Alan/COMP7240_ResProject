import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
# ==== Robust path definitions ====
# data_tool.py -> flaskr/tools/data_tool.py
CURRENT_DIR = os.path.dirname(__file__)                     # .../flaskr/tools
FLASKR_DIR = os.path.dirname(CURRENT_DIR)                   # .../flaskr
PROJECT_ROOT = os.path.dirname(FLASKR_DIR)                  # 项目根目录 7240_ResProject

COMMENTS_PATH = os.path.join(FLASKR_DIR, "static", "ml_data", "comments.csv")
LORA_MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "model", "bert_sentiment_lora")
CORPUS_PATH = os.path.join(FLASKR_DIR, "static", "ml_data", "movie_embeddings_lora.pt")

print("[DEBUG] COMMENTS_PATH:", COMMENTS_PATH)
print("[DEBUG] LORA_MODEL_DIR:", LORA_MODEL_DIR)
print("[DEBUG] CORPUS_PATH:", CORPUS_PATH)
def loadData():
    return getMovies(), getGenre(), getRates()


# movieId,title,year,overview,cover_url,genres
def getMovies():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/movie_info.csv"
    # path = os.path.join(rootPath, 'static', 'ml_data', 'movie_info.csv')
    df = pd.read_csv(path)
    df['genres'] = df.genres.str.split('|')

    return df


# A list of the genres.
def getGenre():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/genre.csv"
    # path = os.path.join(rootPath, 'flaskr', 'ml_data', 'movie_info.csv')
    df = pd.read_csv(path, delimiter="|", names=["name", "id"])
    df.set_index('id')
    return df


# user id, item id, rating, timestamp
def getRates():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/ratings.csv"
    df = pd.read_csv(path, delimiter=",", header=0, names=["userId", "movieId", "rating", "timestamp"])
    df = df.drop(columns='timestamp')
    df = df[['userId', 'movieId', 'rating']]

    return df


# itemID | userID | rating
def ratesFromUser(rates):
    itemID = []
    userID = []
    rating = []

    for rate in rates:
        items = rate.split("|")
        userID.append(int(items[0]))
        itemID.append(int(items[1]))
        rating.append(int(items[2]))

    ratings_dict = {
        "userId": userID,
        "movieId": itemID,
        "rating": rating,
    }

    return pd.DataFrame(ratings_dict)


import time

# ========== Comment Related Functions ==========

def getComments():
    """Get all comments"""
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/comments.csv"
    
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['userId', 'movieId', 'text', 'label', 'timestamp'])
        df.to_csv(path, index=False)
        return df
    
    df = pd.read_csv(path)
    return df


def getCommentsByMovie(movie_id):
    """Get all comments for a movie"""
    df = getComments()
    result = df[df['movieId'] == movie_id].sort_values(by='timestamp', ascending=False)
    return result


def getCommentByUserAndMovie(user_id, movie_id):
    """Get a user's comment for a movie"""
    df = getComments()
    result = df[(df['userId'] == user_id) & (df['movieId'] == movie_id)]
    if len(result) > 0:
        return result.iloc[0].to_dict()
    return None


def addOrUpdateComment(user_id, movie_id, text, label):
    """
    Add or update comment
    If the user has already commented on this movie, update it; otherwise add new comment
    """
    df = getComments()
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/comments.csv"
    
    # Check if already exists
    mask = (df['userId'] == user_id) & (df['movieId'] == movie_id)
    
    if mask.any():
        # Update existing comment
        df.loc[mask, 'text'] = text
        df.loc[mask, 'label'] = label
        df.loc[mask, 'timestamp'] = int(time.time())
        df.to_csv(path, index=False)
        return {'action': 'updated', 'userId': user_id, 'movieId': movie_id}
    else:
        # Add new comment
        new_comment = pd.DataFrame({
            'userId': [user_id],
            'movieId': [movie_id],
            'text': [text],
            'label': [label],
            'timestamp': [int(time.time())]
        })
        
        # If file is empty, write header; otherwise append
        if len(df) == 0:
            new_comment.to_csv(path, index=False)
        else:
            new_comment.to_csv(path, mode='a', header=False, index=False)
        
        return {'action': 'added', 'userId': user_id, 'movieId': movie_id}


def deleteComment(user_id, movie_id):
    """Delete comment"""
    df = getComments()
    df = df[~((df['userId'] == user_id) & (df['movieId'] == movie_id))]
    
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/comments.csv"
    df.to_csv(path, index=False)
    
    return True


def getUserComments(user_id):
    """Get all comments for a user"""
    df = getComments()
    result = df[df['userId'] == user_id].sort_values(by='timestamp', ascending=False)
    return result


# ========== User Rating Related Functions ==========

def getUserRatings(user_id):
    """Get all ratings for a user"""
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/user_ratings.csv"
    
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
        df.to_csv(path, index=False)
        return df
    
    df = pd.read_csv(path)
    result = df[df['userId'] == user_id]
    return result


def saveUserRating(user_id, movie_id, rating):
    """Save or update user rating"""
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/user_ratings.csv"
    
    # Read existing data
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
    
    # Check if already exists
    mask = (df['userId'] == user_id) & (df['movieId'] == movie_id)
    
    if mask.any():
        # Update existing rating
        df.loc[mask, 'rating'] = rating
        df.loc[mask, 'timestamp'] = int(time.time())
    else:
        # Add new rating
        new_rating = pd.DataFrame({
            'userId': [user_id],
            'movieId': [movie_id],
            'rating': [rating],
            'timestamp': [int(time.time())]
        })
        df = pd.concat([df, new_rating], ignore_index=True)
    
    df.to_csv(path, index=False)
    return True


def deleteUserRating(user_id, movie_id):
    """Delete user rating"""
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/user_ratings.csv"
    
    if not os.path.exists(path):
        return True
    
    df = pd.read_csv(path)
    df = df[~((df['userId'] == user_id) & (df['movieId'] == movie_id))]
    df.to_csv(path, index=False)
    return True


# ========== User Like/Dislike Related Functions ==========

def getUserLikesData(user_id):
    """Get all like/dislike records for a user"""
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/user_likes.csv"
    
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['userId', 'movieId', 'action', 'timestamp'])
        df.to_csv(path, index=False)
        return df
    
    df = pd.read_csv(path)
    result = df[df['userId'] == user_id]
    return result


def saveUserLike(user_id, movie_id, action):
    """
    Save user like/dislike
    action: 'like' or 'dislike'
    """
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/user_likes.csv"
    
    # Read existing data
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=['userId', 'movieId', 'action', 'timestamp'])
    
    # Check if already exists
    mask = (df['userId'] == user_id) & (df['movieId'] == movie_id)
    
    if mask.any():
        # Update existing record
        df.loc[mask, 'action'] = action
        df.loc[mask, 'timestamp'] = int(time.time())
    else:
        # Add new record
        new_like = pd.DataFrame({
            'userId': [user_id],
            'movieId': [movie_id],
            'action': [action],
            'timestamp': [int(time.time())]
        })
        df = pd.concat([df, new_like], ignore_index=True)
    
    df.to_csv(path, index=False)
    return True


def deleteUserLike(user_id, movie_id):
    """Delete user like/dislike record"""
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/user_likes.csv"
    
    if not os.path.exists(path):
        return True
    
    df = pd.read_csv(path)
    df = df[~((df['userId'] == user_id) & (df['movieId'] == movie_id))]
    df.to_csv(path, index=False)
    return True


def deleteUserLikesBatch(user_id, movie_ids):
    """Batch delete user like/dislike records"""
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/user_likes.csv"
    
    if not os.path.exists(path):
        return True
    
    df = pd.read_csv(path)
    # Delete specified user's specified movie records
    mask = (df['userId'] == user_id) & (df['movieId'].isin(movie_ids))
    df = df[~mask]
    df.to_csv(path, index=False)
    return True


def deleteAllUserLikes(user_id):
    """Delete all like/dislike records for a user"""
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/user_likes.csv"
    
    if not os.path.exists(path):
        return True
    
    df = pd.read_csv(path)
    df = df[df['userId'] != user_id]
    df.to_csv(path, index=False)
    return True


def deleteUserRatingsBatch(user_id, movie_ids):
    """Batch delete user ratings"""
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/user_ratings.csv"
    
    if not os.path.exists(path):
        return True
    
    df = pd.read_csv(path)
    mask = (df['userId'] == user_id) & (df['movieId'].isin(movie_ids))
    df = df[~mask]
    df.to_csv(path, index=False)
    return True


def deleteAllUserRatings(user_id):
    """Delete all ratings for a user"""
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/user_ratings.csv"
    
    if not os.path.exists(path):
        return True
    
    df = pd.read_csv(path)
    df = df[df['userId'] != user_id]
    df.to_csv(path, index=False)
    return True

# ========== LoRA-based Positive Comment Recommendation ==========

def load_lora_model():
    """Load LoRA-finetuned sentiment model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) base classification model
    base_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
    )

    # 2) load LoRA adapter weights
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
    model.to(device)
    model.eval()

    # 3) tokenizer:
    #    - 如果 LORA_MODEL_DIR 里有 tokenizer（vocab.txt 等），优先用
    #    - 否则退回到 bert-base-uncased 的 tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(LORA_MODEL_DIR)
    except Exception as e:
        print("[WARN] Failed to load tokenizer from LORA_MODEL_DIR, fallback to 'bert-base-uncased'. Error:", e)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer, device



def build_movie_positive_corpus():
    """
    Build positive-review embeddings corpus at movie level,
    using comments.csv where label == 1.
    """
    if not os.path.exists(COMMENTS_PATH):
        return

    df = pd.read_csv(COMMENTS_PATH)
    pos_df = df[df["label"] == 1].reset_index(drop=True)
    if pos_df.empty:
        return

    model, tokenizer, device = load_lora_model()

    texts = pos_df["text"].astype(str).tolist()
    movie_ids = pos_df["movieId"].tolist()

    all_vecs = []
    with torch.no_grad():
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model.base_model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            cls = outputs.last_hidden_state[:, 0, :]  # [B, hidden]
            all_vecs.append(cls.cpu())

    embeddings = torch.cat(all_vecs, dim=0)  # [N, hidden]

    torch.save(
        {
            "embeddings": embeddings,
            "movie_ids": movie_ids,
            "texts": texts,
        },
        CORPUS_PATH,
    )


def recommend_movies_from_text(text, top_k=5):
    """
    Given a free-text comment, recommend movies based on positive-comment corpus.
    Returns: [{'movieId': ..., 'score': ...}, ...]
    """
    # ensure corpus exists
    if not os.path.exists(CORPUS_PATH):
        build_movie_positive_corpus()
        if not os.path.exists(CORPUS_PATH):
            return []

    data = torch.load(CORPUS_PATH)
    corpus_embeddings = data["embeddings"]   # [N, hidden] on CPU
    movie_ids = data["movie_ids"]

    model, tokenizer, device = load_lora_model()

    enc = tokenizer(
        [text],
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

    q_vec = F.normalize(q_vec, p=2, dim=-1)
    c_vec = F.normalize(corpus_embeddings, p=2, dim=-1)
    sims = torch.mm(q_vec.cpu(), c_vec.T).squeeze(0)  # [N]

    top_k = min(top_k, len(sims))
    vals, idxs = torch.topk(sims, k=top_k)

    # collect and deduplicate by movieId
    raw = []
    for score, idx in zip(vals.tolist(), idxs.tolist()):
        raw.append({
            "movieId": int(movie_ids[idx]),
            "score": float(score),
        })

    seen = set()
    result = []
    for r in raw:
        mid = r["movieId"]
        if mid in seen:
            continue
        seen.add(mid)
        result.append(r)
        if len(result) >= top_k:
            break

    return result
