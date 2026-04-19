## This branck is add the industry pipeline for this recommendation program with deepseek rank
## Upgrade the original recommendation system based on single collaborative filtering (KNN) to a three-layer pipeline architecture that meets industry standards: multi recall → big language model ranking → diversity reordering.

用户评分数据
      │
      ▼
┌─────────────────────────────────────┐
│           多路召回 (Recall)          │
│  ┌─────────┐ ┌─────────┐ ┌───────┐ │
│  │ User-CF │ │ Item-CF │ │热门召回│ │
│  └─────────┘ └─────────┘ └───────┘ │
│         合并去重 → 候选池 (~300部)   │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│        DeepSeek LLM 精排 (Ranking)   │
│  - 构建用户画像（高评分电影）          │
│  - 输入候选电影元数据（标题、类型、概述）│
│  - 语义理解 + 逻辑推理 → 个性化排序    │
│  - 输出排序列表 + 推荐理由             │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│         多样性重排 (Re-ranking)       │
│  - 规则：连续同类型电影不超过2部        │
│  - 提升浏览体验，避免审美疲劳           │
└─────────────────────────────────────┘
      │
      ▼
   最终推荐列表（12部）

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
