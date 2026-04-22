## This branck is add the industry pipeline for this recommendation program with deepseek rank
## Upgrade the original recommendation system based on single collaborative filtering (KNN) to a three-layer pipeline architecture that meets industry standards: multi recall → big language model ranking → diversity reordering.
```text
User Rating Data
      │
      ▼
┌─────────────────────────────────────────┐
│          Multi-Channel Recall            │
│  ┌───────────┐ ┌───────────┐ ┌────────┐ │
│  │ User-CF   │ │ Item-CF   │ │Popular │ │
│  └───────────┘ └───────────┘ └────────┘ │
│      Merge & Deduplicate → Candidate Pool│
│               (~300 movies)              │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│      DeepSeek LLM Fine-Ranking           │
│  - Build User Profile (high-rated movies)│
│  - Input Candidate Movie Metadata        │
│    (title, genre, overview)              │
│  - Semantic Understanding + Reasoning    │
│  - Personalized Ranking                  │
│  - Output: Ranked List + Reasons         │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│        Diversity Re-ranking              │
│  - Rule: ≤2 consecutive same genre       │
│  - Improve browsing experience           │
└─────────────────────────────────────────┘
      │
      ▼
   Final Recommendation List (12 movies)
```
## add a new Python lib：
## pip install openai pyyaml

## 如果你要测试大模型精排部分，你需要这个
deepseek:
  api_key: "sk-你的API密钥"
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com"     # 默认 API 地址
  #调整生成参数
  temperature: 0.5
  max_tokens: 500
## 讲此部分放入项目根目录与flaskr并排的config.yaml文件中, 我默认了读取此文件, 读取不到后面的流程会降级到 User-CF 协同过滤排序

关闭多样性
main.py 文件下：
Modify this function
def getRecommendationBy(user_rates):
....
ranked_ids, reasoning = pipeline.rank_with_deepseek(
                candidates_dict, user_rates_df, movies, top_k=12, apply_rerank=True ////此处改为False 关闭多样性重排
            )
....

## change a TF-IDF and Type tags (Multi Hot) mixed similarity recommendation in: 
# Modify this function
def getLikedSimilarBy(user_likes):....
The weighted fusion recommendation list will include works of the same genre as the favorite movies, as well as movies of different genres but related to content, enhancing diversity and interpretability.

## This main branch now is for merge the total program together
## Create an environment

```
conda create -n lab3
conda activate lab3

```

## Install Python packages 

```
pip install --upgrade setuptools wheel pyquery
conda install -c conda-forge scikit-surprise
pip install -r requirements.txt

```

## Run the project
```
flask --app flaskr run --debug
```

## Add the recommendation algorithm
You only need to modify the `main.py` file. Its path is as follows:
```
path: /flaskr/main.py
```

## About the Dataset
The dataset path is: ./flaskr/static/ml_data/

The ratings.csv file includes the following columns:
- userId: the IDs of users.  
- movieId: the IDs of movies.  
- rating: the rating given by the user to the movie, on a 5-star scale
- timestamp: the time when the user rated the movie, recorded in seconds since the epoch (as returned by time(2) function). A larger timestamp means the rating was made later.

You can use pandas to convert the timestamp to standard date and time. For example, 1717665888 corresponds to 2024-06-06 09:24:48.
```
import pandas as pd
timestamp = 1717665888
dt_str = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')
print(dt_str)
```
```

## About the ratings.csv
The ratings.csv file includes the following columns:
- userId: the IDs of users.  
- movieId: the IDs of movies.  
- rating: the rating given by the user to the movie, on a 5-star scale
- timestamp: the time when the user rated the movie, recorded in seconds since the epoch (as returned by time(2) function). A larger timestamp means the rating was made later.

You can use pandas to convert the timestamp to standard date and time. For example, 1717665888 corresponds to 2024-06-06 09:24:48.
```
import pandas as pd
timestamp = 1717665888
dt_str = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')
print(dt_str)
```