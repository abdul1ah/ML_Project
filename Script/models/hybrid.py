import os
import pickle
import numpy as np
import pandas as pd

# -----------------------------------------------
# LOAD COLLABORATIVE MODEL + MOVIE METADATA
# -----------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

with open(os.path.join(BASE_DIR, "trained_collaborative_model.pkl"), "rb") as f:
    collaborative_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "movie_metadata.pkl"), "rb") as f:
    movie_metadata = pickle.load(f)

# Ensure all metadata keys are ints
movie_metadata = {int(k): v for k, v in movie_metadata.items()}

# -----------------------------------------------
# LOAD CONTENT-BASED SIMILARITY MATRIX
# -----------------------------------------------
with open(os.path.join(BASE_DIR, "movie_similarity_matrix.pkl"), "rb") as f:
    similarity_matrix = pickle.load(f)

# -----------------------------------------------
# LOAD RATINGS
# -----------------------------------------------
ratings_df = pd.read_csv(os.path.join(BASE_DIR, "Data", "sampled_data.csv"))
ratings_df["movieId"] = ratings_df["movieId"].astype(int)
ratings_df["userId"] = ratings_df["userId"].astype(int)

all_movie_ids = sorted(list(movie_metadata.keys()))
movie_index_map = {m: i for i, m in enumerate(all_movie_ids)}

# -----------------------------------------------
# CONTENT SCORE CALCULATION (normalized)
# -----------------------------------------------
def content_score(user_id, movie_id):
    """
    Compute normalized content-based score in rating scale [0.5, 5.0].
    """
    user_movies = ratings_df[ratings_df["userId"] == user_id]
    if user_movies.empty:
        return 2.75  # neutral fallback

    if movie_id not in movie_index_map:
        return 2.75

    target_idx = movie_index_map[movie_id]

    score_sum = 0.0
    weight_sum = 0.0

    for _, row in user_movies.iterrows():
        mid = int(row["movieId"])
        if mid not in movie_index_map:
            continue

        idx = movie_index_map[mid]
        sim = similarity_matrix[target_idx][idx]

        score_sum += sim * row["rating"]
        weight_sum += abs(sim)

    if weight_sum == 0:
        return 2.75  # neutral fallback

    raw = score_sum / weight_sum      # raw CB score (~0.5–2 range normally)

    # --- NORMALIZATION ---
    # Clip raw similarity-weighted score (for stability)
    raw = max(min(raw, 1.0), 0.0)

    # Map [0,1] → [0.5,5.0] rating scale
    normalized = 0.5 + 4.5 * raw

    return normalized

# -----------------------------------------------
# HYBRID PREDICTION
# -----------------------------------------------
def hybrid_predict(user_id, movie_id, alpha=0.6):
    user_id = int(user_id)
    movie_id = int(movie_id)

    cb = content_score(user_id, movie_id)
    cf = collaborative_model.predict(uid=user_id, iid=movie_id).est

    return alpha * cf + (1 - alpha) * cb

# -----------------------------------------------
# GET TOP RECOMMENDATIONS
# -----------------------------------------------
def hybrid_recommend(user_id, n=10):
    user_id = int(user_id)
    watched = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
    not_watched = [m for m in all_movie_ids if m not in watched]

    predictions = []
    for m in not_watched:
        score = hybrid_predict(user_id, m)
        predictions.append((m, score))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top = predictions[:n]

    results = []
    for movie_id, pred in top:
        info = movie_metadata.get(movie_id, {
            "title": "Unknown",
            "genres": "N/A",
            "cast_names": "N/A"
        })
        results.append((info["title"], info["genres"], info["cast_names"], pred))

    return results

# -----------------------------------------------
# TEST OUTPUT
# -----------------------------------------------
print("\nTop hybrid recommendations for user 1:\n")
for t, g, c, r in hybrid_recommend(1, 5):
    print(f"Title: {t}, Genres: {g}, Cast: {c}, Predicted Rating: {r:.2f}\n")
