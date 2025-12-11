import os
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "Script", "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# -------------------------------
# Load content-based artifacts
# -------------------------------
with open(os.path.join(SAVED_MODELS_DIR, "movie_similarity_matrix.pkl"), "rb") as f:
    similarity_matrix = pickle.load(f)

with open(os.path.join(SAVED_MODELS_DIR, "content_movie_index.pkl"), "rb") as f:
    movie_index_map = pickle.load(f)

with open(os.path.join(SAVED_MODELS_DIR, "movie_metadata.pkl"), "rb") as f:
    movie_metadata = pickle.load(f)

# ensure all metadata keys are ints
movie_metadata = {int(k): v for k, v in movie_metadata.items()}

# -------------------------------
# Load collaborative model
# -------------------------------
with open(os.path.join(SAVED_MODELS_DIR, "trained_collaborative_model.pkl"), "rb") as f:
    collaborative_model = pickle.load(f)

# -------------------------------
# Load ratings
# -------------------------------
ratings_df = pd.read_csv(os.path.join(BASE_DIR, "Data", "sampled_data.csv"))
ratings_df["movieId"] = ratings_df["movieId"].astype(int)
ratings_df["userId"] = ratings_df["userId"].astype(int)

all_movie_ids = sorted(list(movie_metadata.keys()))
movie_index_map = {m: i for i, m in enumerate(all_movie_ids)}

# -------------------------------
# Content-based score
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

    raw = score_sum / weight_sum
    raw = max(min(raw, 1.0), 0.0)
    normalized = 0.5 + 4.5 * raw
    return normalized

# -------------------------------
# Hybrid prediction
# -------------------------------
def hybrid_predict(user_id, movie_id, alpha=0.6):
    user_id, movie_id = int(user_id), int(movie_id)
    cb = content_score(user_id, movie_id)
    cf = collaborative_model.predict(uid=user_id, iid=movie_id).est
    return alpha * cf + (1 - alpha) * cb

# -------------------------------
# Top recommendations
# -------------------------------
def hybrid_recommend(user_id, n=10):
    user_id = int(user_id)
    watched = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
    not_watched = [m for m in all_movie_ids if m not in watched]

    predictions = [(m, hybrid_predict(user_id, m)) for m in not_watched]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top = predictions[:n]

    results = []
    for movie_id, pred in top:
        info = movie_metadata.get(movie_id, {"title": "Unknown", "genres": "N/A", "cast_names": "N/A"})
        results.append((info["title"], info["genres"], info["cast_names"], pred))

    return results

# -------------------------------
# Optional: save hybrid artifacts (kept for backward compatibility)
# -------------------------------
with open(os.path.join(SAVED_MODELS_DIR, "hybrid_similarity_matrix.pkl"), "wb") as f:
    pickle.dump(similarity_matrix, f)

with open(os.path.join(SAVED_MODELS_DIR, "hybrid_movie_index_map.pkl"), "wb") as f:
    pickle.dump(movie_index_map, f)

with open(os.path.join(SAVED_MODELS_DIR, "hybrid_movie_metadata.pkl"), "wb") as f:
    pickle.dump(movie_metadata, f)

with open(os.path.join(SAVED_MODELS_DIR, "hybrid_cf_model.pkl"), "wb") as f:
    pickle.dump(collaborative_model, f)

print("Hybrid model artifacts saved successfully in Script/saved_models/")

# -------------------------------
# Testing output
# -------------------------------
if __name__ == "__main__":
    print("\nTop hybrid recommendations for user 1:\n")
    for t, g, c, r in hybrid_recommend(1, 5):
        print(f"Title: {t}, Genres: {g}, Cast: {c}, Predicted Rating: {r:.2f}\n")
