import os
import pickle
from fastapi import FastAPI
from typing import List
import numpy as np

app = FastAPI()

# ----------------------------------------------------
# PATHS
# Script/fastapi/main.py → move up twice → project root
# ----------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)                         # Script/fastapi
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
MODEL_DIR = os.path.join(PROJECT_ROOT, "Script", "saved_models")  # NEW FOLDER

# ----------------------------------------------------
# LOAD HYBRID ARTIFACTS
# ----------------------------------------------------
def load_pickle(name):
    path = os.path.join(MODEL_DIR, name)   # <-- UPDATED
    return pickle.load(open(path, "rb"))

similarity_matrix: np.ndarray = load_pickle("hybrid_similarity_matrix.pkl")
movie_index_map: dict = load_pickle("hybrid_movie_index_map.pkl")
movie_metadata: dict = load_pickle("hybrid_movie_metadata.pkl")
collaborative_model = load_pickle("hybrid_cf_model.pkl")

# ----------------------------------------------------
# CONTENT SCORE
# ----------------------------------------------------
def content_score(user_id: int, movie_id: int):
    if movie_id not in movie_index_map:
        return 2.75

    target_idx = movie_index_map[movie_id]

    raw = 0.0
    weight = 0.0

    # Get user’s watched movies from CF's trainset
    user_ratings = (
        collaborative_model.trainset.ur.get(user_id)
        if user_id in collaborative_model.trainset.ur
        else []
    )

    if not user_ratings:
        return 2.75

    for inner_iid, rating in user_ratings:
        true_movie_id = int(collaborative_model.trainset.to_raw_iid(inner_iid))

        if true_movie_id not in movie_index_map:
            continue

        idx = movie_index_map[true_movie_id]
        sim = similarity_matrix[target_idx][idx]

        raw += sim * rating
        weight += abs(sim)

    if weight == 0:
        return 2.75

    raw_score = raw / weight
    raw_score = max(min(raw_score, 1.0), 0.0)

    return 0.5 + 4.5 * raw_score  # normalized → 0.5–5.0


# ----------------------------------------------------
# HYBRID PREDICT
# ----------------------------------------------------
def hybrid_predict(user_id: int, movie_id: int, alpha: float = 0.7):
    cf_pred = collaborative_model.predict(user_id, movie_id).est
    cb_pred = content_score(user_id, movie_id)
    return alpha * cf_pred + (1 - alpha) * cb_pred


# ----------------------------------------------------
# RECOMMENDATION ENDPOINT
# ----------------------------------------------------
@app.get("/recommend")
def recommend(user_id: int, n: int = 10):
    user_id = int(user_id)

    all_movies = list(movie_index_map.keys())

    # Movies user has already rated
    watched_inner = collaborative_model.trainset.ur.get(user_id, [])
    watched_raw = {
        int(collaborative_model.trainset.to_raw_iid(iid))
        for iid, _ in watched_inner
    }

    unseen = [m for m in all_movies if m not in watched_raw]

    preds = []
    for m in unseen:
        score = hybrid_predict(user_id, m)
        meta = movie_metadata.get(m, {})
        preds.append({
            "movie_id": m,
            "title": meta.get("title", "Unknown"),
            "genres": meta.get("genres", "N/A"),
            "cast": meta.get("cast_names", "N/A"),
            "predicted_rating": round(score, 3)
        })

    preds.sort(key=lambda x: x["predicted_rating"], reverse=True)
    return preds[:n]


# ----------------------------------------------------
# Run using:
# uvicorn main:app --reload
# ----------------------------------------------------
