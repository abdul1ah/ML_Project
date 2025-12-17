import os
import pandas as pd
import json
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV, train_test_split
from surprise.accuracy import rmse, mae

# -------------------------------
# PATH LOGIC (Sync with Backend, Hybrid, and Content-Based)
# -------------------------------
PROJECT_ROOT = os.getenv("GITHUB_WORKSPACE") 
if not PROJECT_ROOT:
    # Local fallback: Script/models/collaborative.py -> ../../ -> Root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "Script", "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

df_path = os.getenv('DATA_PATH') or os.path.join(DATA_DIR, "sampled_data.csv")
mapping_path = os.path.join(DATA_DIR, "movie_mapping.csv")

print(f"DEBUG: Project Root: {PROJECT_ROOT}")
print(f"DEBUG: Saving models to: {SAVED_MODELS_DIR}")

# -------------------------------
# Load datasets
# -------------------------------
df = pd.read_csv(df_path)
mapping_df = pd.read_csv(mapping_path)

# -------------------------------
# Process Movie Metadata
# -------------------------------
mapping_df['genres'] = mapping_df['genres'].fillna("N/A")
mapping_df['cast'] = mapping_df['cast'].fillna("[]")

def extract_cast_names(cast_str, top_n=5):
    try:
        cast_list = json.loads(cast_str)
        names = [c['name'] for c in cast_list]
        return ', '.join(names[:top_n])
    except Exception:
        return "N/A"

mapping_df['cast_names'] = mapping_df['cast'].apply(extract_cast_names)

# Create movie_info dictionary (using movieId as int key)
movie_info = mapping_df.set_index('movieId')[['title', 'genres', 'cast_names']].to_dict(orient='index')
movie_info = {int(k): v for k, v in movie_info.items()}

# Ensure all movies in ratings are in movie_info
for mid in df['movieId'].unique():
    mid_int = int(mid)
    if mid_int not in movie_info:
        movie_info[mid_int] = {'title': 'Unknown', 'genres': 'N/A', 'cast_names': 'N/A'}

# -------------------------------
# Train SVD Model
# -------------------------------
reader = Reader(rating_scale=(df.rating.min(), df.rating.max()))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Hyperparameter tuning
param_grid = {
    'n_factors': [50, 100],
    'n_epochs': [20, 50],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.05]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

# Final Training on full set
best_model = SVD(**gs.best_params['rmse'])
trainset = data.build_full_trainset()
best_model.fit(trainset)

print(f"\nBest RMSE: {gs.best_score['rmse']:.4f}")

# -------------------------------
# Save Artifacts (ALIGNED WITH BACKEND)
# -------------------------------
def save_pickle(obj, filename):
    path = os.path.join(SAVED_MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# 1. The Collaborative Model
save_pickle(best_model, "trained_collaborative_model.pkl")

# 2. Metadata (using the 'hybrid' prefix for backend consistency)
save_pickle(movie_info, "hybrid_movie_metadata.pkl")

print(f"SUCCESS: Collaborative artifacts saved in {SAVED_MODELS_DIR}")

# -------------------------------
# Local Test Recommendation
# -------------------------------
def quick_test(user_id=1):
    all_movies = df['movieId'].unique()
    # Predict for first 5 movies
    preds = [(m, best_model.predict(uid=user_id, iid=m).est) for m in all_movies[:5]]
    print(f"\nSample Predictions for User {user_id}:")
    for mid, score in preds:
        title = movie_info.get(int(mid), {}).get('title', 'Unknown')
        print(f"- {title}: {score:.2f}")

if __name__ == "__main__":
    quick_test()