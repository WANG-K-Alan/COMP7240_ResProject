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

# Helper function to get current user ID
def get_current_user_id():
    """Get current user ID from Cookie, default is 611"""
    user_id = request.cookies.get('current_user_id', '611')
    return int(user_id)


@bp.route('/', methods=('GET', 'POST'))
def index():
    """Home page - Recommendation page"""
    default_genres = genres.to_dict('records')
    user_genres = request.cookies.get('user_genres')
    if user_genres:
        user_genres = user_genres.split(",")
    else:
        # Default select "All" (id=0)
        user_genres = ['0']
    
    # Get current user ID
    current_user_id = get_current_user_id()
    
    # Load user data from CSV if Cookie is empty
    user_rates = request.cookies.get('user_rates')
    if user_rates:
        user_rates = user_rates.split(",")
    else:
        # Load from CSV
        user_ratings_df = getUserRatings(current_user_id)
        if len(user_ratings_df) > 0:
            user_rates = []
            for _, row in user_ratings_df.iterrows():
                user_rates.append(f"{int(row['userId'])}|{int(row['movieId'])}|{int(row['rating'])}|0")
        else:
            user_rates = []
    
    user_likes = request.cookies.get('user_likes')
    user_dislikes = request.cookies.get('user_dislikes')
    if user_likes:
        user_likes = user_likes.split(",")
    else:
        user_likes = []
    if user_dislikes:
        user_dislikes = user_dislikes.split(",")
    else:
        user_dislikes = []
    
    # If Cookie is empty, load from CSV
    if not user_likes and not user_dislikes:
        user_likes_df = getUserLikesData(current_user_id)
        if len(user_likes_df) > 0:
            likes_list = user_likes_df[user_likes_df['action'] == 'like']['movieId'].tolist()
            dislikes_list = user_likes_df[user_likes_df['action'] == 'dislike']['movieId'].tolist()
            user_likes = [str(id) for id in likes_list]
            user_dislikes = [str(id) for id in dislikes_list]
    
    
    # Get movies by selected genres (for rating popup)
    default_genres_movies = getMoviesByGenres(user_genres)[:10]
    
    # Get recommendations (requires ratings) - not limited by genres
    recommendations_movies, recommendations_message = getRecommendationBy(user_rates)
    
    # Get similar movies and likes list - not limited by genres
    likes_similar_movies, likes_similar_message = getLikedSimilarBy([int(numeric_string) for numeric_string in user_likes], [int(numeric_string) for numeric_string in user_dislikes])
    likes_movies = getUserLikesBy(user_likes)
    
    # If user selected specific genres (not All), filter recommendation results
    # But the filtering here is "expansion" not "restriction": show recommendations of selected genres
    if '0' not in user_genres:
        # Get user selected genre names
        genres_mask = genres['id'].isin([int(id) for id in user_genres])
        user_genres_list = [1 if has is True else 0 for has in genres_mask]
        user_genres_df = pd.DataFrame(user_genres_list, columns=['value'])
        user_genres_df = pd.concat([user_genres_df, genres['name']], axis=1)
        interested_genres = user_genres_df[user_genres_df['value'] == 1]['name'].tolist()
        interested_genres = [g for g in interested_genres if g != 'All']
        
        if len(interested_genres) > 0:
            # Filter recommendation results: only show movies containing selected genres
            if recommendations_movies:
                recommendations_movies = [m for m in recommendations_movies if is_genre_match(m['genres'], interested_genres)]
            
            # Filter similar movies: only show movies containing selected genres
            if likes_similar_movies:
                likes_similar_movies = [m for m in likes_similar_movies if is_genre_match(m['genres'], interested_genres)]
            
            # Don't filter likes list, because these are movies user already liked, should always show
    
    # Limit display to first 12 movies (after genres filtering)
    if recommendations_movies and len(recommendations_movies) > 12:
        recommendations_movies = recommendations_movies[:12]
    
    if likes_similar_movies and len(likes_similar_movies) > 12:
        likes_similar_movies = likes_similar_movies[:12]

    return render_template('index.html',
                           genres=default_genres,
                           user_genres=user_genres,
                           user_rates=user_rates,
                           user_likes=user_likes,
                           user_dislikes=user_dislikes,
                           current_user_id=current_user_id,
                           default_genres_movies=default_genres_movies,
                           recommendations=recommendations_movies,
                           recommendations_message=recommendations_message,
                           likes_similars=likes_similar_movies,
                           likes_similar_message=likes_similar_message,
                           likes=likes_movies,
                           )


@bp.route('/browse', methods=('GET',))
def browse():
    """Browse page - Display all movies, support filtering by genre, search and pagination"""
    try:
        # Get current user ID
        current_user_id = get_current_user_id()
        
        # Get user Cookie
        user_likes = request.cookies.get('user_likes')
        user_dislikes = request.cookies.get('user_dislikes')
        if user_likes:
            user_likes = user_likes.split(",")
        else:
            user_likes = []
        if user_dislikes:
            user_dislikes = user_dislikes.split(",")
        else:
            user_dislikes = []
        
        # If Cookie is empty, load from CSV
        if not user_likes and not user_dislikes:
            user_likes_df = getUserLikesData(current_user_id)
            if len(user_likes_df) > 0:
                likes_list = user_likes_df[user_likes_df['action'] == 'like']['movieId'].tolist()
                dislikes_list = user_likes_df[user_likes_df['action'] == 'dislike']['movieId'].tolist()
                user_likes = [str(id) for id in likes_list]
                user_dislikes = [str(id) for id in dislikes_list]
        
        # Get filter parameters
        selected_genre = request.args.get('genre', '')
        search_query = request.args.get('search', '').strip()
        
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = 48  # Display 48 movies per page
        
        # Get all genres
        all_genres = genres.to_dict('records')
        
        print(f"[DEBUG] Total movies: {len(movies)}")
        print(f"[DEBUG] Selected genre: {selected_genre}")
        print(f"[DEBUG] Search query: {search_query}")
        print(f"[DEBUG] Page: {page}")
        
        # Filter movies by genre
        if selected_genre and selected_genre != 'all':
            # Filter movies containing specified genre
            filtered_movies = movies[movies['genres'].apply(lambda x: selected_genre in x)]
            filter_message = f"Showing movies in genre: {selected_genre}"
        else:
            # Show all movies
            filtered_movies = movies
            filter_message = f"Showing all movies"
        
        print(f"[DEBUG] After genre filter: {len(filtered_movies)} movies")
        
        # Filter by search keyword
        if search_query:
            def search_match(row):
                # Search title
                if search_query.lower() in str(row['title']).lower():
                    return True
                # Search description
                if pd.notna(row['overview']) and isinstance(row['overview'], str):
                    if search_query.lower() in row['overview'].lower():
                        return True
                # Search genres
                if any(search_query.lower() in genre.lower() for genre in row['genres']):
                    return True
                return False
            
            filtered_movies = filtered_movies[filtered_movies.apply(search_match, axis=1)]
            
            if selected_genre and selected_genre != 'all':
                filter_message = f"Search results for '{search_query}' in genre: {selected_genre}"
            else:
                filter_message = f"Search results for '{search_query}'"
        
        print(f"[DEBUG] After search filter: {len(filtered_movies)} movies")
        
        # Calculate pagination
        total_count = len(filtered_movies)
        
        # If no movies, set default values
        if total_count == 0:
            total_pages = 1
            page = 1
            display_movies = []
        else:
            total_pages = (total_count + per_page - 1) // per_page  # Round up
            
            # Ensure page number is within valid range
            if page < 1:
                page = 1
            elif page > total_pages:
                page = total_pages
            
            # Get movies for current page
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_movies = filtered_movies.iloc[start_idx:end_idx]
            display_movies = paginated_movies.to_dict('records')
        
        print(f"[DEBUG] Displaying {len(display_movies)} movies on page {page}/{total_pages}")
        
        return render_template('browse.html',
                               all_movies=display_movies,
                               all_genres=all_genres,
                               selected_genre=selected_genre,
                               search_query=search_query,
                               filter_message=filter_message,
                               user_likes=user_likes,
                               user_dislikes=user_dislikes,
                               current_user_id=get_current_user_id(),
                               total_count=total_count,
                               current_page=page,
                               total_pages=total_pages,
                               per_page=per_page
                               )
    except Exception as e:
        print(f"[ERROR] Browse route error: {e}")
        import traceback
        traceback.print_exc()
        raise


@bp.route('/my-likes', methods=('GET',))
def my_likes():
    """My Likes page - Display all liked and disliked records, support batch deletion"""
    user_id = get_current_user_id()
    
    # Load user data from CSV file
    likes_data = getUserLikesData(user_id)
    
    user_likes = []
    user_dislikes = []
    
    for _, row in likes_data.iterrows():
        movie_id = str(int(row['movieId']))
        if row['action'] == 'like':
            user_likes.append(movie_id)
        elif row['action'] == 'dislike':
            user_dislikes.append(movie_id)
    
    # Get pagination parameters and tab parameter
    page = request.args.get('page', 1, type=int)
    tab = request.args.get('tab', 'likes')  # 'likes' or 'dislikes'
    per_page = 48
    
    # Select movies to display based on tab
    if tab == 'dislikes':
        selected_ids = user_dislikes
    else:
        selected_ids = user_likes
    
    # Get all selected movies (in order, newest first)
    all_movies_list = []
    if len(selected_ids) > 0:
        mask = movies['movieId'].isin([int(movieId) for movieId in selected_ids])
        results = movies.loc[mask]
        
        # Sort by selection order (newest first)
        for _id in selected_ids:
            movie = results.loc[results['movieId'] == int(_id)]
            if len(movie) > 0:
                all_movies_list.append(movie.iloc[0].to_dict())
    
    # Calculate pagination
    total_count = len(all_movies_list)
    
    if total_count == 0:
        total_pages = 1
        page = 1
        display_movies = []
    else:
        total_pages = (total_count + per_page - 1) // per_page
        
        if page < 1:
            page = 1
        elif page > total_pages:
            page = total_pages
        
        # Get movies for current page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        display_movies = all_movies_list[start_idx:end_idx]
    
    return render_template('my_likes.html',
                           liked_movies=display_movies,
                           total_count=total_count,
                           current_page=page,
                           total_pages=total_pages,
                           per_page=per_page,
                           current_tab=tab,
                           likes_count=len(user_likes),
                           dislikes_count=len(user_dislikes),
                           user_likes=','.join(user_likes),
                           user_dislikes=','.join(user_dislikes),
                           current_user_id=get_current_user_id()
                           )


def getUserLikesBy(user_likes, limit=20):
    """Get user liked movies, return in like order (newest first), limit quantity"""
    results = []

    if len(user_likes) > 0:
        mask = movies['movieId'].isin([int(movieId) for movieId in user_likes])
        results = movies.loc[mask]

        # Sort by user_likes order (newest liked first)
        original_orders = pd.DataFrame()
        for _id in user_likes:
            movie = results.loc[results['movieId'] == int(_id)]
            if len(original_orders) == 0:
                original_orders = movie
            else:
                original_orders = pd.concat([movie, original_orders])
        results = original_orders
        
        # Limit return quantity (most recent N movies)
        if len(results) > limit:
            results = results.head(limit)

    if len(results) > 0:
        return results.to_dict('records')
    return results

def is_genre_match(movie_genres, interested_genres):
    return bool(set(movie_genres).intersection(set(interested_genres)))

def getMoviesByGenres(user_genres):
    """Filter movies by user selected genres"""
    results = []
    
    # If no genres selected, return empty
    if len(user_genres) == 0:
        return results
    
    # If "All" is selected (id=0), return all movies
    if '0' in user_genres:
        return movies.to_dict('records')
    
    # Otherwise filter by selected genres
    genres_mask = genres['id'].isin([int(id) for id in user_genres])
    user_genres_list = [1 if has is True else 0 for has in genres_mask]
    user_genres_df = pd.DataFrame(user_genres_list, columns=['value'])
    user_genres_df = pd.concat([user_genres_df, genres['name']], axis=1)
    interested_genres = user_genres_df[user_genres_df['value'] == 1]['name'].tolist()
    
    # Filter out "All"
    interested_genres = [g for g in interested_genres if g != 'All']
    
    if len(interested_genres) > 0:
        results = movies[movies['genres'].apply(lambda x: is_genre_match(x, interested_genres))]
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
def getLikedSimilarBy(user_likes, user_dislikes):
    """
    根据用户的喜欢和不喜欢电影，综合类型标签和剧情文本相似度，推荐相似电影。
    返回 (results, message) 元组。
    """
    results = []
    message = "No similar movies found."
    
    if len(user_likes) == 0:
        return results, "Please like some movies first to get similar recommendations."

    # ========== 1. 类型标签相似度计算 ==========
    genre_matrix, genre_df, feature_list = item_representation_based_movie_genres(movies)
    
    # 构建用户画像（喜欢 - 不喜欢）
    user_profile_genre = build_user_profile(
        liked_movieIds=user_likes,
        disliked_movieIds=user_dislikes if user_dislikes else [],
        item_rep_vector=genre_df,
        feature_list=feature_list,
        normalized=True
    )
    
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
    beta  = 0.6   # TF‑IDF 文本权重

    # 标准化到 [0,1]
    genre_sim_norm = (genre_similarities - genre_similarities.min()) / (genre_similarities.max() - genre_similarities.min() + 1e-8)
    text_sim_norm  = (text_similarities - text_similarities.min()) / (text_similarities.max() - text_similarities.min() + 1e-8)

    combined_similarities = alpha * genre_sim_norm + beta * text_sim_norm

    # ========== 4. 排序并排除已喜欢/不喜欢电影 ==========
    exclude_ids = set(user_likes) | set(user_dislikes)  # 排除已标记的电影
    sim_df = pd.DataFrame({
        'movieId': movies['movieId'],
        'similarity': combined_similarities
    })
    sim_df = sim_df[~sim_df['movieId'].isin(exclude_ids)]
    top_movie_ids = sim_df.sort_values('similarity', ascending=False).head(12)['movieId'].tolist()
    results = movies[movies['movieId'].isin(top_movie_ids)]

    if len(results) > 0:
        message = "These movies are recommended based on genre and plot similarity to your liked movies."
    
    return results.to_dict('records'), message


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
def build_user_profile(liked_movieIds, disliked_movieIds, item_rep_vector, feature_list, normalized=True):

    liked_df = item_rep_vector[item_rep_vector['movieId'].isin(liked_movieIds)]
    if len(liked_df) > 0:
        liked_profile = liked_df[feature_list].mean()
    else:
        liked_profile = pd.Series([0]*len(feature_list), index=feature_list)
    
    disliked_df = item_rep_vector[item_rep_vector['movieId'].isin(disliked_movieIds)]
    if len(disliked_df) > 0:
        disliked_profile = disliked_df[feature_list].mean()
    else:
        disliked_profile = pd.Series([0]*len(feature_list), index=feature_list)
    
    user_profile = liked_profile - disliked_profile
    user_profile = user_profile.clip(lower=0)
    
    if normalized and user_profile.sum() > 0:
        user_profile = user_profile / user_profile.sum()
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


# ========== Comment API Routes ==========

@bp.route('/api/movie/<int:movie_id>/comments', methods=['GET'])
def get_movie_comments(movie_id):
    """Get all comments for a movie"""
    try:
        comments = getCommentsByMovie(movie_id)
        return {
            'success': True,
            'count': len(comments),
            'comments': comments.to_dict('records')
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}


@bp.route('/api/movie/<int:movie_id>/my-comment', methods=['GET'])
def get_my_comment(movie_id):
    """Get current user's comment for a movie"""
    try:
        user_id = get_current_user_id()
        comment = getCommentByUserAndMovie(user_id, movie_id)
        return {
            'success': True,
            'comment': comment
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}


@bp.route('/api/movie/<int:movie_id>/comment', methods=['POST'])
def add_or_update_comment(movie_id):
    """Add or update comment"""
    try:
        text = request.form.get('text', '').strip()
        label = int(request.form.get('label', 1))
        
        if not text:
            return {'success': False, 'message': 'Comment content cannot be empty'}
        
        if len(text) > 1000:
            return {'success': False, 'message': 'Comment content cannot exceed 1000 characters'}
        
        user_id = get_current_user_id()
        result = addOrUpdateComment(user_id, movie_id, text, label)
        
        return {
            'success': True,
            'action': result['action'],
            'message': 'Comment updated' if result['action'] == 'updated' else 'Comment added'
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}


@bp.route('/api/movie/<int:movie_id>/comment', methods=['DELETE'])
def delete_my_comment(movie_id):
    """Delete current user's comment"""
    try:
        user_id = get_current_user_id()
        deleteComment(user_id, movie_id)
        return {
            'success': True,
            'message': 'Comment deleted'
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}


# ========== User Rating API ==========

@bp.route('/api/user/ratings', methods=['GET'])
def get_user_ratings():
    """Get all ratings for current user"""
    try:
        user_id = get_current_user_id()
        ratings = getUserRatings(user_id)
        
        # Convert to frontend format: "userId|movieId|rating"
        ratings_list = []
        for _, row in ratings.iterrows():
            ratings_list.append(f"{int(row['userId'])}|{int(row['movieId'])}|{int(row['rating'])}|0")
        
        return {
            'success': True,
            'ratings': ratings_list
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}


@bp.route('/api/user/rating', methods=['POST'])
def save_user_rating():
    """Save user rating"""
    try:
        user_id = get_current_user_id()
        movie_id = int(request.form.get('movieId'))
        rating = int(request.form.get('rating'))
        
        saveUserRating(user_id, movie_id, rating)
        
        return {
            'success': True,
            'message': 'Rating saved'
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}


@bp.route('/api/user/rating/<int:movie_id>', methods=['DELETE'])
def delete_user_rating_route(movie_id):
    """Delete user rating"""
    try:
        user_id = get_current_user_id()
        deleteUserRating(user_id, movie_id)
        
        return {
            'success': True,
            'message': 'Rating deleted'
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}


# ========== User Like/Dislike API ==========

@bp.route('/api/user/likes', methods=['GET'])
def get_user_likes():
    """Get all like/dislike records for current user"""
    try:
        user_id = get_current_user_id()
        likes_data = getUserLikesData(user_id)
        
        likes = []
        dislikes = []
        
        for _, row in likes_data.iterrows():
            movie_id = str(int(row['movieId']))
            if row['action'] == 'like':
                likes.append(movie_id)
            elif row['action'] == 'dislike':
                dislikes.append(movie_id)
        
        return {
            'success': True,
            'likes': likes,
            'dislikes': dislikes
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}


@bp.route('/api/user/like', methods=['POST'])
def save_user_like():
    """Save user like/dislike"""
    try:
        user_id = get_current_user_id()
        movie_id = int(request.form.get('movieId'))
        action = request.form.get('action')  # 'like' or 'dislike' or 'remove'
        
        if action == 'remove':
            deleteUserLike(user_id, movie_id)
        else:
            saveUserLike(user_id, movie_id, action)
        
        return {
            'success': True,
            'message': 'Operation successful'
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}


@bp.route('/api/user/likes/batch-delete', methods=['POST'])
def batch_delete_user_likes():
    """Batch delete user like/dislike records"""
    try:
        user_id = get_current_user_id()
        movie_ids = request.json.get('movieIds', [])
        
        if not movie_ids:
            return {'success': False, 'message': 'No movies selected for deletion'}
        
        # Batch delete like/dislike records
        deleteUserLikesBatch(user_id, movie_ids)
        
        # Batch delete rating records
        deleteUserRatingsBatch(user_id, movie_ids)
        
        return {
            'success': True,
            'message': f'Deleted {len(movie_ids)} records'
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}


@bp.route('/api/user/likes/delete-all', methods=['POST'])
def delete_all_user_likes():
    """Delete all like/dislike and rating records for user"""
    try:
        user_id = get_current_user_id()
        
        # Delete all like/dislike records
        deleteAllUserLikes(user_id)
        
        # Delete all rating records
        deleteAllUserRatings(user_id)
        
        return {
            'success': True,
            'message': 'All records deleted'
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}


# ========== Comment-based Recommendation API (LoRA) ==========
@bp.route('/api/recommend-from-comment', methods=['POST'])
def recommend_from_comment():
    """
    Given a free-text comment, recommend movies based on
    positive comments + LoRA-BERT embeddings.
    """
    try:
        # 尽量从 JSON 里取；如果 Content-Type 不标准，就强制解析一次
        data = request.get_json(silent=True)
        if data is None:
            # 尝试从原始数据再解析一次 JSON
            try:
                import json
                raw = request.data.decode('utf-8') if request.data else ''
                data = json.loads(raw) if raw else {}
            except Exception:
                data = {}

        # 如果还是空，退回到 form/query
        if not data:
            data = request.form.to_dict() or request.args.to_dict()

        text = (data.get('text') or '').strip()
        top_k = int(data.get('top_k', 5))

        if not text:
            return {'success': False, 'message': 'Comment text cannot be empty.'}, 400

        # 调用 data_tool.py 中的 LoRA 推荐函数
        recs = recommend_movies_from_text(text, top_k=top_k)  # [{'movieId', 'score'}, ...]

        movie_map = movies.set_index('movieId')
        result_movies = []
        for r in recs:
            mid = r['movieId']
            if mid in movie_map.index:
                m = movie_map.loc[mid].to_dict()
                m['movieId'] = int(mid)
                m['score'] = r['score']
                result_movies.append(m)

        return {'success': True, 'movies': result_movies}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {'success': False, 'message': str(e)}, 500
