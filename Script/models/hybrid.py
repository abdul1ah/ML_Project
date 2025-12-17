import os
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# PATH LOGIC (Sync with Backend)
# -------------------------------
# Check if running in CI (GitHub Actions)
PROJECT_ROOT = os.getenv("GITHUB_WORKSPACE") 
if not PROJECT_ROOT:
    # Local: Script/models/hybrid.py -> ../../ -> Root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "Script", "saved_models")
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

print(f"DEBUG: Project Root: {PROJECT_ROOT}")
print(f"DEBUG: Working with models in: {SAVED_MODELS_DIR}")

# -------------------------------
# Load content-based artifacts
# -------------------------------
def load_pickle(name):
    path = os.path.join(SAVED_MODELS_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)

# These names must match what content_based.py saves
similarity_matrix = load_pickle("hybrid_similarity_matrix.pkl")
movie_index_map = load_pickle("hybrid_movie_index_map.pkl")
movie_metadata = load_pickle("hybrid_movie_metadata.pkl")

# Ensure all metadata keys are ints
movie_metadata = {int(k): v for k, v in movie_metadata.items()}

# -------------------------------
# Load collaborative model
# -------------------------------
# This name must match what collaborative.py saves
collaborative_model = load_pickle("trained_collaborative_model.pkl")

# -------------------------------
# Load ratings
# -------------------------------
ratings_df = pd.read_csv(os.path.join(DATA_DIR, "sampled_data.csv"))
ratings_df["movieId"] = ratings_df["movieId"].astype(int)
ratings_df["userId"] = ratings_df["userId"].astype(int)

all_movie_ids = sorted(list(movie_metadata.keys()))
movie_index_map = {m: i for i, m in enumerate(all_movie_ids)}

# -------------------------------
# Scoring Logic
# -------------------------------
def content_score(user_id, movie_id):
    user_movies = ratings_df[ratings_df["userId"] == user_id]
    if user_movies.empty or movie_id not in movie_index_map:
        return 2.75

    target_idx = movie_index_map[movie_id]
    score_sum, weight_sum = 0.0, 0.0

    for _, row in user_movies.iterrows():
        mid = int(row["movieId"])
        if mid not in movie_index_map:
            continue
        idx = movie_index_map[mid]
        sim = similarity_matrix[target_idx][idx]
        score_sum += sim * row["rating"]
        weight_sum += abs(sim)

    if weight_sum == 0:
        return 2.75

    raw = np.clip(score_sum / weight_sum, 0.0, 1.0)
    return 0.5 + 4.5 * raw

def hybrid_predict(user_id, movie_id, alpha=0.6):
    user_id, movie_id = int(user_id), int(movie_id)
    cb = content_score(user_id, movie_id)
    cf = collaborative_model.predict(uid=user_id, iid=movie_id).est
    return alpha * cf + (1 - alpha) * cb

def hybrid_recommend(user_id, n=10):
    user_id = int(user_id)
    watched = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
    not_watched = [m for m in all_movie_ids if m not in watched]

    predictions = [(m, hybrid_predict(user_id, m)) for m in not_watched]
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for movie_id, pred in predictions[:n]:
        info = movie_metadata.get(movie_id, {"title": "Unknown", "genres": "N/A", "cast_names": "N/A"})
        results.append((info["title"], info["genres"], info["cast_names"], pred))
    return results

# -------------------------------
# Final Save (The Artifacts Backend Needs)
# -------------------------------
def save_pickle(obj, name):
    path = os.path.join(SAVED_MODELS_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

save_pickle(similarity_matrix, "hybrid_similarity_matrix.pkl")
save_pickle(movie_index_map, "hybrid_movie_index_map.pkl")
save_pickle(movie_metadata, "hybrid_movie_metadata.pkl")
save_pickle(collaborative_model, "trained_collaborative_model.pkl")

print(f"SUCCESS: Hybrid assembly complete. All artifacts saved in {SAVED_MODELS_DIR}")

if __name__ == "__main__":
    print("\nTesting Recommendations for User 1:")
    for t, g, c, r in hybrid_recommend(1, 5):
        print(f"- {t} ({r:.2f})")