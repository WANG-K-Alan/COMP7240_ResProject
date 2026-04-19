import pandas as pd
from surprise import Reader, Dataset, KNNWithMeans
import os
import json
import yaml
from openai import OpenAI

# ------------------- 基于用户的协同过滤召回 ------------------
def recall_user_cf(user_rates_df, all_rates_df, all_movies_df, n=200):
    """
    基于用户的协同过滤召回
    参数：
        user_rates_df : 当前用户的评分 DataFrame (userId, movieId, rating)
        all_rates_df  : 全局评分数据 (userId, movieId, rating)
        all_movies_df : 电影元数据
        n             : 召回数量
    返回：
        list of tuples (movieId, predicted_rating)
    """
    if len(user_rates_df) == 0:
        return []
    
    # 合并当前用户评分与全局评分
    training_rates = pd.concat([all_rates_df, user_rates_df], ignore_index=True)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(training_rates[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    algo = KNNWithMeans(sim_options={'name': 'pearson', 'user_based': True})
    algo.fit(trainset)
    
    user_id = 611  # 固定当前用户ID
    rated_movie_ids = user_rates_df['movieId'].tolist()
    all_movie_ids = all_movies_df['movieId'].unique()
    
    predictions = []
    for mid in all_movie_ids:
        if mid not in rated_movie_ids:
            pred = algo.predict(user_id, mid)
            predictions.append((mid, pred.est))
    
    # 按预测评分降序排序，取前n个
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

# ------------------- 基于物品的协同过滤召回 ------------------
def recall_item_cf(user_rates_df, all_rates_df, all_movies_df, n=200):
    """
    基于物品的协同过滤召回
    """
    if len(user_rates_df) == 0:
        return []
    
    training_rates = pd.concat([all_rates_df, user_rates_df], ignore_index=True)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(training_rates[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    # 关键：user_based=False 即基于物品
    algo = KNNWithMeans(sim_options={'name': 'pearson', 'user_based': False})
    algo.fit(trainset)
    
    user_id = 611
    # 获取用户评分过的电影作为“种子”
    rated_movies = user_rates_df['movieId'].tolist()
    all_movie_ids = all_movies_df['movieId'].unique()
    
    # 对于每个种子电影，找出与其最相似的 k 个电影（这里取 k=20）
    candidates = set()
    for seed in rated_movies:
        try:
            # 获取相似物品的内部ID
            inner_id = algo.trainset.to_inner_iid(seed)
            neighbors = algo.get_neighbors(inner_id, k=20)
            for n_inner in neighbors:
                raw_id = algo.trainset.to_raw_iid(n_inner)
                if raw_id not in rated_movies:
                    candidates.add(raw_id)
        except ValueError:
            continue
    
    # 由于 KNNWithMeans 的 get_neighbors 不返回相似度分数，我们简单返回候选列表，分数置为0（后续精排会重新计算）
    return [(mid, 0.0) for mid in candidates]

# ------------------- 热门召回 ------------------
def recall_popular(all_rates_df, all_movies_df, n=50):
    """
    热门召回：基于平均评分和评分人数（可选）
    这里简化：仅按平均评分降序取前n部电影
    """
    # 计算每部电影的平均评分
    movie_stats = all_rates_df.groupby('movieId')['rating'].agg(['mean', 'count'])
    # 过滤掉评分人数过少的电影（例如少于10人）
    popular_movies = movie_stats[movie_stats['count'] >= 10].sort_values('mean', ascending=False)
    top_movie_ids = popular_movies.head(n).index.tolist()
    # 分数为平均评分
    return [(mid, popular_movies.loc[mid, 'mean']) for mid in top_movie_ids]

# ------------------- 多路召回整合 ------------------
def multi_recall(user_rates_df, all_rates_df, all_movies_df):
    """
    执行多路召回，返回候选电影ID集合及对应的原始分数字典
    """
    recall_results = {}
    
    # 1. User-CF 召回
    user_cf = recall_user_cf(user_rates_df, all_rates_df, all_movies_df, n=200)
    for mid, score in user_cf:
        recall_results[mid] = {'user_cf_score': score}
    
    # 2. Item-CF 召回
    item_cf = recall_item_cf(user_rates_df, all_rates_df, all_movies_df, n=200)
    for mid, score in item_cf:
        if mid not in recall_results:
            recall_results[mid] = {}
        recall_results[mid]['item_cf_score'] = score
    
    # 3. 热门召回
    popular = recall_popular(all_rates_df, all_movies_df, n=50)
    for mid, score in popular:
        if mid not in recall_results:
            recall_results[mid] = {}
        recall_results[mid]['popular_score'] = score
    
    return recall_results

# ------------------- 配置加载 ------------------
def load_config():
    """
    加载项目根目录下的 config.yaml 文件
    返回配置字典
    """
    # 获取项目根目录路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "config.yaml 文件未找到。如需使用 DeepSeek 精排，请在项目根目录创建 config.yaml 并配置 API Key。\n"
            "若仅测试功能，可忽略此提示，系统将自动回退至 User-CF 协同过滤排序。"
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# ------------------- DeepSeek 精排 ------------------
def rank_with_deepseek(candidates_dict, user_rates_df, movies_df, top_k=12, apply_rerank=True):
    """
    使用 DeepSeek API 对候选电影进行智能精排
    
    参数:
        candidates_dict: 多路召回返回的候选字典 {movieId: {'user_cf_score': ..., ...}}
        user_rates_df: 用户评分DataFrame
        movies_df: 电影元数据DataFrame
        top_k: 最终返回的电影数量
        apply_rerank: 是否在精排后应用多样性重排
    
    返回:
        (ranked_movie_ids, reasoning): 排序后的movieId列表和排序理由
    """
    # 如果没有候选电影，直接返回空列表
    if not candidates_dict:
        return [], "No candidates to rank."
    
    # ---------- 1. 加载配置 ----------
    config = load_config()
    api_key = config['deepseek']['api_key']
    model = config['deepseek'].get('model', 'deepseek-chat')
    
    # ---------- 2. 构建用户画像 ----------
    high_rated = user_rates_df[user_rates_df['rating'] >= 4]['movieId'].tolist()
    high_rated_movies = movies_df[movies_df['movieId'].isin(high_rated)]
    
    user_profile = ""
    if len(high_rated_movies) > 0:
        user_profile = "user high-rated movies:\n"
        for _, row in high_rated_movies.head(5).iterrows():
            genres_str = ', '.join(row['genres'][:3])
            user_profile += f"- {row['title']} (type: {genres_str})\n"
    else:
        rated_movies = movies_df[movies_df['movieId'].isin(user_rates_df['movieId'].tolist())]
        if len(rated_movies) > 0:
            user_profile = "user rated movies:\n"
            for _, row in rated_movies.head(5).iterrows():
                genres_str = ', '.join(row['genres'][:3])
                user_profile += f"- {row['title']} (type: {genres_str})\n"
    
    # ---------- 3. 构建候选电影列表 ----------
    candidate_ids = list(candidates_dict.keys())
    # 限制候选数量，避免超过模型上下文（建议不超过50个）
    if len(candidate_ids) > 50:
        # 按 user_cf 分数取前50个
        sorted_ids = sorted(candidate_ids,
                           key=lambda x: candidates_dict[x].get('user_cf_score', 0),
                           reverse=True)[:50]
        candidate_ids = sorted_ids
    
    candidate_list = ""
    candidate_movies = movies_df[movies_df['movieId'].isin(candidate_ids)]
    for _, row in candidate_movies.iterrows():
        genres_str = ', '.join(row['genres'][:3])
        overview = row['overview']
        if isinstance(overview, str):
            overview = overview[:200] + "..." if len(overview) > 200 else overview
        else:
            overview = "No overview available"
        candidate_list += f"ID:{row['movieId']} | {row['title']} | type:{genres_str} | overview:{overview}\n"
    
    # ---------- 4. 构建 Prompt ----------
    system_prompt = """You are a movie recommendation assistant. Based on the user's profile and the candidate movie list, you need to select the top K movies that best match the user's preferences. Consider factors such as genre similarity, user ratings, and diversity of recommendations.

Requirements:
1. Ensure that the recommended movies closely align with the user's high-rated movies, prioritizing those with similar genres and themes.
2. Ensure the recommendation list is diverse, avoiding consecutive recommendations of the same movie type.
3. The final output format must be strictly JSON, with the following structure:
{
  "ranked_ids": [1, 2, 3, ...],
  "reasoning": "Brief explanation of your ranking logic (no more than 100 words)."
}

Attention:- Do not include any content outside the JSON format in your response."""
    
    user_prompt = f"""{user_profile} candidate movies: {candidate_list} Please select the {top_k} most recommended movies from the above list and provide the sorted movieId list in English."""
    
    # ---------- 5. 调用 DeepSeek API ----------
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        ranked_ids = result.get("ranked_ids", [])
        reasoning = result.get("reasoning", "No reasoning provided.")
        
        # 打印 Token 使用情况
        print(f"[DeepSeek] Model: {model}, Input tokens: {response.usage.prompt_tokens}, "
              f"Output tokens: {response.usage.completion_tokens}")
        
        # ---------- 6. 多样性重排 ----------
        if apply_rerank and len(ranked_ids) >= top_k:
            # 只对前 top_k 个进行重排（重排函数会在阶段四实现）
            to_rerank = ranked_ids[:top_k]
            reranked = rerank_diversity(to_rerank, movies_df, max_same_genre=2)
            ranked_ids = reranked + ranked_ids[top_k:]
        
        return ranked_ids[:top_k], reasoning
        
    except Exception as e:
        print(f"[DeepSeek Error] {e}")
        # 降级方案：回退到 user_cf 分数排序
        print("[DeepSeek] Falling back to user_cf score ranking...")
        sorted_ids = sorted(candidate_ids,
                           key=lambda x: candidates_dict[x].get('user_cf_score', 0),
                           reverse=True)[:top_k]
        return sorted_ids, "Fallback: API调用失败，使用协同过滤分数排序。"

# ------------------- 多样性重排函数 ------------------
def rerank_diversity(ranked_ids, movies_df, max_same_genre=2):
    """
    对排序后的电影列表进行多样性重排
    规则：连续同主要类型的电影不超过 max_same_genre 部
    
    参数:
        ranked_ids: 已排序的电影ID列表
        movies_df: 电影元数据
        max_same_genre: 最多允许连续出现几部同类型电影
    
    返回:
        重排后的movieId列表
    """
    if len(ranked_ids) <= 3:
        return ranked_ids
    
    def get_primary_genre(mid):
        row = movies_df[movies_df['movieId'] == mid]
        if len(row) == 0:
            return 'Unknown'
        genres = row.iloc[0]['genres']
        return genres[0] if isinstance(genres, list) and len(genres) > 0 else 'Unknown'
    
    result = []
    remaining = list(ranked_ids)
    
    while remaining:
        found = False
        # 获取最近 (max_same_genre) 部已选电影的主要类型
        recent_genres = [get_primary_genre(mid) for mid in result[-max_same_genre:]]
        
        for i, mid in enumerate(remaining):
            genre = get_primary_genre(mid)
            if recent_genres.count(genre) < max_same_genre:
                result.append(remaining.pop(i))
                found = True
                break
        
        # 如果找不到满足条件的，直接取第一个
        if not found and remaining:
            result.append(remaining.pop(0))
    
    return result