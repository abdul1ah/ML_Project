import os
import pandas as pd
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

# -------------------------------
# PATH LOGIC (Sync with Backend & CI)
# -------------------------------
PROJECT_ROOT = os.getenv("GITHUB_WORKSPACE") 
if not PROJECT_ROOT:
    # Local: Script/models/content_based.py -> ../../ -> Root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "Script", "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Define file paths
df_path = os.getenv('DATA_PATH') or os.path.join(DATA_DIR, "sampled_data.csv")
mapping_path = os.path.join(DATA_DIR, "movie_mapping.csv")

print(f"DEBUG: Project Root: {PROJECT_ROOT}")
print(f"DEBUG: Loading data from: {df_path}")

# -------------------------------
# Load Datasets
# -------------------------------
ratings_df = pd.read_csv(df_path)
mapping_df = pd.read_csv(mapping_path)

mapping_df['genres'] = mapping_df['genres'].fillna("N/A")
mapping_df['cast'] = mapping_df['cast'].fillna("[]")

# Extract top 5 actors
def extract_cast_names(cast_str, top_n=5):
    try:
        cast_list = json.loads(cast_str)
        names = [c['name'] for c in cast_list]
        return ', '.join(names[:top_n])
    except Exception:
        return ""

mapping_df['cast_names'] = mapping_df['cast'].apply(extract_cast_names)

# Create combined feature string for TF-IDF
mapping_df['features'] = mapping_df['genres'].str.replace('|', ' ', regex=False) + ' ' + mapping_df['cast_names']

# Build movie metadata dictionary
movie_metadata = {}
for _, row in mapping_df.iterrows():
    movie_metadata[int(row['movieId'])] = {
        'title': row.get('title', 'Unknown'),
        'genres': row['genres'],
        'cast_names': row['cast_names']
    }

# -------------------------------
# TF-IDF & Similarity Computation
# -------------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(mapping_df['features'])
movie_ids = mapping_df['movieId'].values
movie_index = {int(mid): idx for idx, mid in enumerate(movie_ids)}

def compute_movie_similarity(matrix):
    return cosine_similarity(matrix)

similarity_matrix = compute_movie_similarity(tfidf_matrix)

# -------------------------------
# Prediction Function (for Evaluation)
# -------------------------------
def predict_rating(user_id, movie_id):
    if movie_id not in movie_index:
        return np.nan
    
    idx = movie_index[movie_id]
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    
    if user_ratings.empty:
        return np.nan

    sim_sum = 0.0
    weighted_sum = 0.0

    for _, row in user_ratings.iterrows():
        rated_movie_id = int(row['movieId'])
        if rated_movie_id not in movie_index:
            continue
        rated_idx = movie_index[rated_movie_id]
        sim = similarity_matrix[idx, rated_idx]
        weighted_sum += sim * row['rating']
        sim_sum += sim

    return weighted_sum / sim_sum if sim_sum > 0 else user_ratings['rating'].mean()

# -------------------------------
# Evaluation
# -------------------------------
reader = Reader(rating_scale=(ratings_df.rating.min(), ratings_df.rating.max()))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

predictions = []
for uid, iid, true_r in testset:
    pred_r = predict_rating(uid, iid)
    if np.isnan(pred_r):
        pred_r = ratings_df['rating'].mean()
    predictions.append((true_r, pred_r))

true_ratings, pred_ratings = zip(*predictions)
rmse_val = np.sqrt(np.mean((np.array(true_ratings) - np.array(pred_ratings))**2))
print(f"Content-Based RMSE: {rmse_val:.4f}")

# -------------------------------
# Save Artifacts (ALIGNED WITH BACKEND)
# -------------------------------
def save_pickle(obj, filename):
    with open(os.path.join(SAVED_MODELS_DIR, filename), "wb") as f:
        pickle.dump(obj, f)

save_pickle(similarity_matrix, "hybrid_similarity_matrix.pkl")
save_pickle(movie_index, "hybrid_movie_index_map.pkl")
save_pickle(movie_metadata, "hybrid_movie_metadata.pkl")
save_pickle(ratings_df, "content_ratings_df.pkl")

print(f"SUCCESS: Content-based artifacts saved to {SAVED_MODELS_DIR}")