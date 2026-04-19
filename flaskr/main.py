from . import pipeline

from flask import (
    Blueprint, render_template, request
)

from .tools.data_tool import *

from surprise import Reader
from surprise import KNNBasic, KNNWithMeans
from surprise import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

bp = Blueprint('main', __name__, url_prefix='/')

movies, genres, rates = loadData()


@bp.route('/', methods=('GET', 'POST'))
def index():
    default_genres = genres.to_dict('records')
    user_genres = request.cookies.get('user_genres')
    if user_genres:
        user_genres = user_genres.split(",")
    else:
        user_genres = []
    user_rates = request.cookies.get('user_rates')
    if user_rates:
        user_rates = user_rates.split(",")
    else:
        user_rates = []
    user_likes = request.cookies.get('user_likes')
    if user_likes:
        user_likes = user_likes.split(",")
    else:
        user_likes = []
    default_genres_movies = getMoviesByGenres(user_genres)[:10]
    recommendations_movies, recommendations_message = getRecommendationBy(user_rates)
    likes_similar_movies, likes_similar_message = getLikedSimilarBy([int(numeric_string) for numeric_string in user_likes])
    likes_movies = getUserLikesBy(user_likes)

    return render_template('index.html',
                           genres=default_genres,
                           user_genres=user_genres,
                           user_rates=user_rates,
                           user_likes=user_likes,
                           default_genres_movies=default_genres_movies,
                           recommendations=recommendations_movies,
                           recommendations_message=recommendations_message,
                           likes_similars=likes_similar_movies,
                           likes_similar_message=likes_similar_message,
                           likes=likes_movies,
                           )


def getUserLikesBy(user_likes):
    results = []

    if len(user_likes) > 0:
        mask = movies['movieId'].isin([int(movieId) for movieId in user_likes])
        results = movies.loc[mask]

        original_orders = pd.DataFrame()
        for _id in user_likes:
            movie = results.loc[results['movieId'] == int(_id)]
            if len(original_orders) == 0:
                original_orders = movie
            else:
                original_orders = pd.concat([movie, original_orders])
        results = original_orders

    if len(results) > 0:
        return results.to_dict('records')
    return results

def is_genre_match(movie_genres, interested_genres):
    return bool(set(movie_genres).intersection(set(interested_genres)))

def getMoviesByGenres(user_genres):
    results = []
    if len(user_genres) > 0:
        genres_mask = genres['id'].isin([int(id) for id in user_genres])
        user_genres = [1 if has is True else 0 for has in genres_mask]
        user_genres_df = pd.DataFrame(user_genres,columns=['value'])
        user_genres_df = pd.concat([user_genres_df, genres['name']], axis=1)
        interested_genres = user_genres_df[user_genres_df['value'] == 1]['name'].tolist()
        results = movies[movies['genres'].apply(lambda x: is_genre_match(x, interested_genres))]

    if len(results) > 0:
        return results.to_dict('records')
    return results

# Modify this function
def getRecommendationBy(user_rates):
    results = []
    message = "No recommendations."
    
    if len(user_rates) > 0:
        user_rates_df = ratesFromUser(user_rates)
        user_rates_df['userId'] = 611
        
        # 多路召回
        candidates_dict = pipeline.multi_recall(user_rates_df, rates, movies)
        
        if candidates_dict:
            # DeepSeek 精排
            ranked_ids, reasoning = pipeline.rank_with_deepseek(
                candidates_dict, user_rates_df, movies, top_k=12, apply_rerank=True
            )
            
            if ranked_ids:
                results = movies[movies['movieId'].isin(ranked_ids)]
                # 保持 DeepSeek 返回的顺序
                results = results.set_index('movieId').loc[ranked_ids].reset_index()
                message = f"DeepSeek AI Recommendation: {reasoning}"
    
    if len(results) > 0:
        return results.to_dict('records'), message
    return results, message


# Modify this function
def getLikedSimilarBy(user_likes):
    results = []
    if len(user_likes) > 0:
        # ========== 1. 类型标签相似度计算 ==========
        # 原有的 Multi-Hot 表示函数
        genre_matrix, genre_df, feature_list = item_representation_based_movie_genres(movies)
        user_profile_genre = build_user_profile(user_likes, genre_df, feature_list, weighted=False, normalized=True)
        genre_similarities = cosine_similarity([user_profile_genre.values], genre_matrix).flatten()
        
        # ========== 2. TF‑IDF 文本相似度计算 ==========
        corpus = movies['overview'].fillna('').tolist()
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        liked_indices = movies[movies['movieId'].isin(user_likes)].index.tolist()
        if not liked_indices:
            return [], "No liked movies found."
        
        user_profile_text = tfidf_matrix[liked_indices].mean(axis=0)
        user_profile_text = np.asarray(user_profile_text).reshape(1, -1)
        text_similarities = cosine_similarity(user_profile_text, tfidf_matrix).flatten()
        
        # ========== 3. 加权融合 ==========
        alpha = 0.4   # 类型标签权重
        beta = 0.6    # TF‑IDF 文本权重
        
        # 将两种相似度标准化到 [0,1] 区间
        genre_sim_norm = (genre_similarities - genre_similarities.min()) / (genre_similarities.max() - genre_similarities.min() + 1e-8)
        text_sim_norm = (text_similarities - text_similarities.min()) / (text_similarities.max() - text_similarities.min() + 1e-8)
        
        combined_similarities = alpha * genre_sim_norm + beta * text_sim_norm
        
        # ========== 4. 排序并排除已喜欢电影 ==========
        sim_df = pd.DataFrame({
            'movieId': movies['movieId'],
            'similarity': combined_similarities
        })
        sim_df = sim_df[~sim_df['movieId'].isin(user_likes)]
        top_movie_ids = sim_df.sort_values('similarity', ascending=False).head(12)['movieId'].tolist()
        results = movies[movies['movieId'].isin(top_movie_ids)]
    
    if len(results) > 0:
        return results.to_dict('records'), "These movies are recommended based on genre and plot similarity to your liked movies."
    return results, "No similar movies found."


# Step 1: Representing items with multi-hot vectors
def item_representation_based_movie_genres(movies_df):
    movies_with_genres = movies_df.copy(deep=True)
    genre_list = []
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            movies_with_genres.at[index, genre] = 1
            if genre not in genre_list:
                genre_list.append(genre)

    movies_with_genres = movies_with_genres.fillna(0)

    movies_genre_matrix = movies_with_genres[genre_list].to_numpy()
    
    return movies_genre_matrix, movies_with_genres, genre_list

# Step 2: Building user profile
def build_user_profile(movieIds, item_rep_vector, feature_list, weighted=True, normalized=True):
    user_movie_rating_df = item_rep_vector[item_rep_vector['movieId'].isin(movieIds)]
    user_movie_df = user_movie_rating_df[feature_list].mean()
    user_profile = user_movie_df.T
    
    if normalized:
        user_profile = user_profile / sum(user_profile.values)
        
    return user_profile
# Step 3: Predicting user preference for items
def generate_recommendation_results(user_profile,item_rep_matrix, movies_data, k=12):
    u_v = user_profile.values
    u_v_matrix =  [u_v]
    recommendation_table =  cosine_similarity(u_v_matrix,item_rep_matrix)
    recommendation_table_df = movies_data.copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[:k]
    return rec_result
