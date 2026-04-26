# Movie Recommendation System – COMP7240 Group Project

An intelligent movie recommender system upgraded from single‑algorithm collaborative filtering to a **three‑layer industry pipeline**:  
**Multi‑Channel Recall → DeepSeek LLM Fine‑Ranking → Diversity Re‑ranking**.  
Also includes an enhanced hybrid content‑based engine, BERT semantic search, TF‑IDF retrieval, multi‑user support, persistent CSV storage, and a cyberpunk UI.

---

## Key Features

- **Three‑Stage Recommendation Pipeline**  
  1. **Multi‑Channel Recall** – User‑CF, Item‑CF, and Popularity recall combined into ~300 candidates.  
  2. **LLM Fine‑Ranking** – DeepSeek (or any OpenAI‑compatible model) re‑ranks candidates with semantic understanding and provides human‑readable reasons.  
  3. **Diversity Re‑ranking** – Enforces genre variation (≤2 consecutive same‑genre films); can be disabled.

- **Hybrid Content‑Based “Liked Similar”** – Weighted fusion of TF‑IDF overview similarity and multi‑hot genre similarity. Recommends both same‑genre and thematically related movies for better diversity and explainability.

- **Fallback Strategy** – If the LLM is unreachable or `config.yaml` is missing, the system automatically falls back to User‑CF (KNN) ranking.

- **BERT Semantic & TF‑IDF Search** – Comment‑based BERT recommendations and keyword search across movies.

- **Full‑featured UI** – Cyberpunk theme, glassmorphism cards, multi‑user switching, comments, ratings, likes, batch operations.

---

## Recommendation Pipeline

### 1. Multi‑Channel Recall
Three independent recall methods run in parallel, results merged and deduplicated:

| Channel   | Method                                              | Output                       |
|-----------|-----------------------------------------------------|------------------------------|
| User‑CF   | Surprise KNNWithMeans, Pearson correlation          | Top‑100 predicted            |
| Item‑CF   | Item‑item cosine similarity on rating matrix        | Top‑K similar to highly‑rated|
| Popular   | Global high‑average movies (min votes)              | Always popular               |

Candidate pool size: **~300 movies**

**Condition:** User must have ≥10 ratings for User‑CF; system uses available channels.

### 2. DeepSeek LLM Fine‑Ranking
Candidates are sent to a large language model (DeepSeek) with:
- User profile summary (high‑rated movies)
- Candidate metadata (title, genres, overview)

The LLM returns a **personalised, ranked list** with a **short reason** for each movie.  
**API**: OpenAI‑compatible endpoint configured in `config.yaml`.

### 3. Diversity Re‑ranking
A rule‑based post‑processing step that re‑orders the list so **no more than two consecutive movies belong to the same primary genre**.

**To disable:** In `flaskr/main.py`, inside `getRecommendationBy()`, set `apply_rerank=False`.

### Fallback
If `config.yaml` is missing or the LLM call fails, the system automatically uses **User‑CF (KNN) ranking** instead.

---

## Other Recommendation Methods

### Hybrid Content‑Based (Liked Similar)
*Function:* `getLikedSimilarBy(user_likes)`

**Weighted fusion** of:
- **TF‑IDF cosine similarity** on movie overviews (semantic content)
- **Multi‑hot genre similarity** (Tanimoto coefficient)

Default weight: 0.5 each.  
This ensures recommendations include **both** same‑genre favourites **and** thematically related movies from different genres, improving diversity and interpretability (“because it shares themes with your liked films”).

### BERT Semantic (from comment)
User writes a comment → BERT + LoRA model encodes it → compares with pre‑computed positive‑comment embeddings → returns similar movies. Triggered by the “Get recommendations from this comment” button in movie details.

### TF‑IDF Search (Browse page)
`TfidfVectorizer` (max_features=5000, 1‑2 grams) on overviews. Supports keyword search with genre filters and pagination.

---

## System Architecture

```
User Browser
    ↕ HTTP, Cookies
Flask Application
    ├── Multi‑Recall (User‑CF, Item‑CF, Popular)
    ├── DeepSeek LLM Ranker (or Fallback User‑CF)
    ├── Diversity Re‑ranker
    ├── Hybrid Content Engine
    ├── BERT / TF‑IDF modules
    └── CSV + Cookie Persistence
```

---

## User Interface

Same cyberpunk sci‑fi theme:
- **Home** – collaborative + content‑based sections, genre filters.
- **Browse** – paginated grid, TF‑IDF search, genre dropdown.
- **My Likes** – likes/dislikes tabs, batch delete.
- **Movie Sidebar** – full info, rating, comments, BERT recommendation.

---

## API Endpoints (unchanged)

| Method            | Endpoint                              | Description                 |
|-------------------|---------------------------------------|-----------------------------|
| GET/POST/DELETE   | `/api/user/ratings`                   | Manage ratings              |
| GET/POST          | `/api/user/likes`                     | Manage likes                |
| POST              | `/api/user/likes/batch-delete`        | Batch delete likes          |
| POST              | `/api/user/likes/delete-all`          | Delete all user data        |
| GET/POST/DELETE   | `/api/movie/<id>/comment`             | Manage comments             |
| POST              | `/api/recommend-from-comment`         | BERT‑based recommendation   |

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- pip
- (Optional) NVIDIA GPU for BERT

### Steps
```bash
git clone <repo-url>
cd movie-recommender
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)
```
Flask==3.0.0
pandas==2.1.3
scikit-learn==1.3.2
scikit-surprise==1.1.3
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
openai>=1.0.0
pyyaml>=6.0
```

---

## Configuration

### LLM Fine‑Ranking (DeepSeek)
Create `config.yaml` in the **project root** (next to `flaskr/`):

```yaml
deepseek:
  api_key: "sk-your-api-key"
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com"
  temperature: 0.5
  max_tokens: 500
```

- **api_key**: Your DeepSeek API key.
- **model**: Model name (e.g., `deepseek-chat`).
- **base_url**: API endpoint (can be changed for other OpenAI‑compatible services).
- **temperature / max_tokens**: Control generation randomness and length.

**Fallback:** If this file is missing or the API fails, the system falls back to **User‑CF (KNN)** ranking – the home page will still work.

### Disable Diversity Re‑ranking
In `flaskr/main.py`, locate `getRecommendationBy()` and change the parameter:

```python
ranked_ids, reasoning = pipeline.rank_with_deepseek(
    candidates_dict, user_rates_df, movies, top_k=12, apply_rerank=False
)
```

---

## Running the System

```bash
flask --app flaskr run --debug
# Open http://localhost:5000
```

**Default users:** 611–620 (default 611). No password required.  
All data stored in `flaskr/static/ml_data/`.

---

## Project Structure (after upgrade)

```
movie-recommender/
├── config.yaml                  # LLM API configuration
├── flaskr/
│   ├── __init__.py
│   ├── main.py                  # routes & recommendation logic
│   ├── pipeline.py              # multi-recall, LLM ranking, diversity
│   ├── templates/
│   ├── static/
│   │   ├── css/sci-fi-style.css
│   │   ├── ml_data/*.csv
│   │   └── img/
│   └── tools/data_tool.py
├── requirements.txt
└── README.md
```

---

## Evaluation

A/B tests with ≥12 participants compared:
1. Original single‑algorithm vs upgraded pipeline (survey).
2. Original UI vs cyberpunk UI (survey).

Significant improvements were observed. Full details in the project report.

---

## Contributors

| Name           | Student ID | Role                               |
|----------------|------------|------------------------------------|
| [WANG ZEYU]    | [25400770] | Backend & pipeline                 |
| [TszHin YUEN]  | [25423738] | Frontend & UI design               |
| [DONG WENHUI]  | [25400568] | BERT model & evaluation            |

*COMP7240 Group Project – Hong Kong Baptist University*

---

## Acknowledgements

- Surprise, scikit‑learn, Hugging Face Transformers, DeepSeek, Bulma, Font Awesome.