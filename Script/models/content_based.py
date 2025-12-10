import os
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
import numpy as np

# Load user ratings
df_path = os.getenv('DATA_PATH')
if df_path is None:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    df_path = os.path.join(BASE_DIR, "Data", "sampled_data.csv")

ratings_df = pd.read_csv(df_path)
print("Ratings dataset shape:", ratings_df.shape)

# Load movie mapping
mapping_df = pd.read_csv(os.path.join(BASE_DIR, "Data", "movie_mapping.csv"))
mapping_df['genres'] = mapping_df['genres'].fillna("N/A")
mapping_df['cast'] = mapping_df['cast'].fillna("[]")

# Extract top 5 actors
def extract_cast_names(cast_str, top_n=5):
    try:
        cast_list = json.loads(cast_str)
        names = [c['name'] for c in cast_list]
        return ' '.join(names[:top_n])
    except Exception:
        return ""

mapping_df['cast_names'] = mapping_df['cast'].apply(extract_cast_names)

# Create a combined feature string for each movie
mapping_df['features'] = mapping_df['genres'].str.replace('|', ' ') + ' ' + mapping_df['cast_names']

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(mapping_df['features'])
movie_ids = mapping_df['movieId'].values
movie_index = {mid: idx for idx, mid in enumerate(movie_ids)}

# -------------------------------
# Precompute full movie similarity matrix
# -------------------------------
def compute_movie_similarity(tfidf_matrix):
    """
    Computes the cosine similarity between all movies.
    Returns a square similarity matrix.
    """
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix

# Compute similarity matrix once
similarity_matrix = compute_movie_similarity(tfidf_matrix)

# -------------------------------
# Predict ratings using precomputed similarity
# -------------------------------
def predict_rating(user_id, movie_id):
    if movie_id not in movie_index:
        return np.nan  # unknown movie

    idx = movie_index[movie_id]

    # Movies the user has already rated
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    if user_ratings.empty:
        return np.nan  # unknown user

    sim_sum = 0.0
    weighted_sum = 0.0

    for _, row in user_ratings.iterrows():
        rated_movie_id = row['movieId']
        if rated_movie_id not in movie_index:
            continue
        rated_idx = movie_index[rated_movie_id]
        sim = similarity_matrix[idx, rated_idx]  # use precomputed similarity
        weighted_sum += sim * row['rating']
        sim_sum += sim

    return weighted_sum / sim_sum if sim_sum > 0 else user_ratings['rating'].mean()

# -------------------------------
# Split ratings into train/test for evaluation
# -------------------------------
reader = Reader(rating_scale=(ratings_df.rating.min(), ratings_df.rating.max()))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# -------------------------------
# Evaluate on test set
# -------------------------------
predictions = []
for uid, iid, true_r in testset:
    pred_r = predict_rating(uid, iid)
    if np.isnan(pred_r):
        pred_r = ratings_df['rating'].mean()  # fallback
    predictions.append((true_r, pred_r))

true_ratings, pred_ratings = zip(*predictions)
rmse_val = np.sqrt(np.mean((np.array(true_ratings) - np.array(pred_ratings))**2))
mae_val = np.mean(np.abs(np.array(true_ratings) - np.array(pred_ratings)))

print(f"Content-Based Model Evaluation -> RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")
