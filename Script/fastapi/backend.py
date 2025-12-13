import os
import pickle
import requests
from fastapi import FastAPI, HTTPException, Query
import numpy as np

# app setup
app = FastAPI(
    title="Film Fanatic",
    description="a movie recommendation system that combines different recommendor models to provide personalized movie suggestions to users.",
    version="1.0.0"
)

# paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(BASE_DIR, "Script", "saved_models")

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# load artifacts (once)
def load_pickle(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise RuntimeError(f"missing artifact: {name}")
    with open(path, "rb") as f:
        return pickle.load(f)

similarity_matrix: np.ndarray = load_pickle("hybrid_similarity_matrix.pkl")
movie_index_map: dict = load_pickle("hybrid_movie_index_map.pkl")
movie_metadata: dict = load_pickle("hybrid_movie_metadata.pkl")
collaborative_model = load_pickle("trained_collaborative_model.pkl")

# ensure metadata keys are ints
movie_metadata = {int(k): v for k, v in movie_metadata.items()}

ALL_MOVIES = list(movie_index_map.keys())

# tmdb utilities
def tmdb_search_movie(title: str):
    if not TMDB_API_KEY:
        return {}
    r = requests.get(
        f"{TMDB_BASE_URL}/search/movie",
        params={"api_key": TMDB_API_KEY, "query": title}
    )
    if r.status_code != 200 or not r.json().get("results"):
        return {}
    return r.json()["results"][0]

def enrich_movie(movie_id: int):
    meta = movie_metadata.get(movie_id, {})
    title = meta.get("title", "")
    tmdb_data = tmdb_search_movie(title)
    return {
        "movie_id": movie_id,
        "title": title,
        "genres": meta.get("genres", "N/A"),
        "cast": meta.get("cast_names", "N/A"),
        "overview": tmdb_data.get("overview"),
        "poster": f"https://image.tmdb.org/t/p/w500{tmdb_data['poster_path']}" if tmdb_data.get("poster_path") else None,
        "backdrop": f"https://image.tmdb.org/t/p/w780{tmdb_data['backdrop_path']}" if tmdb_data.get("backdrop_path") else None,
        "release_date": tmdb_data.get("release_date"),
        "rating_tmdb": tmdb_data.get("vote_average")
    }

# content score
def content_score(user_id: int, movie_id: int):
    if movie_id not in movie_index_map:
        return 2.75
    target_idx = movie_index_map[movie_id]
    user_ratings = collaborative_model.trainset.ur.get(user_id)
    if not user_ratings:
        return 2.75
    score_sum, weight_sum = 0.0, 0.0
    for inner_iid, rating in user_ratings:
        raw_id = int(collaborative_model.trainset.to_raw_iid(inner_iid))
        if raw_id not in movie_index_map:
            continue
        idx = movie_index_map[raw_id]
        sim = similarity_matrix[target_idx][idx]
        score_sum += sim * rating
        weight_sum += abs(sim)
    if weight_sum == 0:
        return 2.75
    raw = score_sum / weight_sum
    raw = max(min(raw, 1.0), 0.0)
    return 0.5 + 4.5 * raw

# hybrid predict
def hybrid_predict(user_id: int, movie_id: int, alpha: float):
    cf = collaborative_model.predict(user_id, movie_id).est
    cb = content_score(user_id, movie_id)
    return alpha * cf + (1 - alpha) * cb

# health check
@app.get("/")
def health():
    return {
        "status": "ok",
        "models_loaded": True,
        "total_movies": len(ALL_MOVIES)
    }

# hybrid recommendations (enrich always true)
@app.get("/recommend")
def recommend(user_id: int, n: int = Query(10, le=50), alpha: float = Query(0.7, ge=0.0, le=1.0)):
    user_id = int(user_id)
    watched = {
        int(collaborative_model.trainset.to_raw_iid(iid))
        for iid, _ in collaborative_model.trainset.ur.get(user_id, [])
    }
    candidates = [m for m in ALL_MOVIES if m not in watched]
    scores = [(m, hybrid_predict(user_id, m, alpha)) for m in candidates]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:n]
    results = []
    for mid, score in top:
        data = enrich_movie(mid)  # always enrich
        data["predicted_rating"] = round(score, 3)
        results.append(data)
    return results

# similar movies (content-only)
@app.get("/similar")
def similar_movies(movie_id: int, n: int = 10):
    if movie_id not in movie_index_map:
        raise HTTPException(404, "movie not found")
    idx = movie_index_map[movie_id]
    sims = similarity_matrix[idx]
    pairs = list(enumerate(sims))
    pairs.sort(key=lambda x: x[1], reverse=True)
    results = []
    for i, sim in pairs[1:n+1]:
        mid = list(movie_index_map.keys())[i]
        data = enrich_movie(mid)
        data["similarity"] = round(float(sim), 3)
        results.append(data)
    return results

# search movies
@app.get("/search")
def search_movies(query: str, limit: int = 20):
    query = query.lower()
    matches = []
    for mid, meta in movie_metadata.items():
        if query in meta.get("title", "").lower():
            matches.append(enrich_movie(mid))
        if len(matches) >= limit:
            break
    return matches

# trending movies (uses local metadata too)

@app.get("/trending")
def trending_movies(limit: int = 10):
    if not TMDB_API_KEY:
        raise HTTPException(500, "tmdb api key not configured")

    # fetch trending movies from tmdb
    r = requests.get(
        f"{TMDB_BASE_URL}/trending/movie/week",
        params={"api_key": TMDB_API_KEY}
    )

    if r.status_code != 200:
        raise HTTPException(500, "tmdb error")

    tmdb_movies = r.json().get("results", [])[:limit]

    results = []
    for m in tmdb_movies:
        # find matching movie_id in local metadata by title
        local_movie_id = None
        for mid, meta in movie_metadata.items():
            if meta.get("title", "").lower() == m.get("title", "").lower():
                local_movie_id = mid
                break

        if local_movie_id:
            meta = movie_metadata[local_movie_id]
        else:
            meta = {"genres": "N/A", "cast_names": "N/A"}

        results.append({
            "movie_id": local_movie_id,
            "title": m.get("title"),
            "overview": m.get("overview"),
            "genres": meta.get("genres", "N/A"),
            "cast": meta.get("cast_names", "N/A"),
            "poster": (
                f"https://image.tmdb.org/t/p/w500{m['poster_path']}"
                if m.get("poster_path") else None
            ),
            "backdrop": (
                f"https://image.tmdb.org/t/p/w780{m['backdrop_path']}"
                if m.get("backdrop_path") else None
            ),
            "release_date": m.get("release_date"),
            "rating_tmdb": m.get("vote_average")
        })

    return results

# genre based recommendations (enrich always true)
@app.get("/recommend/genre")
def recommend_by_genre(genre: str, n: int = 10):
    genre = genre.lower()
    candidates = [mid for mid, meta in movie_metadata.items() if genre in meta.get("genres", "").lower()]
    if not candidates:
        return []
    scores = []
    for mid in candidates:
        idx = movie_index_map.get(mid)
        if idx is None:
            continue
        score = similarity_matrix[idx].mean()
        scores.append((mid, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:n]
    results = []
    for mid, score in top:
        data = enrich_movie(mid)  # always enrich
        data["score"] = round(float(score), 3)
        results.append(data)
    return results

# user history
@app.get("/user/history")
def user_history(user_id: int):
    user_id = int(user_id)
    if user_id not in collaborative_model.trainset.ur:
        return []
    history = []
    for inner_iid, rating in collaborative_model.trainset.ur[user_id]:
        movie_id = int(collaborative_model.trainset.to_raw_iid(inner_iid))
        meta = movie_metadata.get(movie_id, {})
        history.append({
            "movie_id": movie_id,
            "title": meta.get("title", "Unknown"),
            "genres": meta.get("genres", "N/A"),
            "cast": meta.get("cast_names", "N/A"),
            "rating": rating
        })
    return history

# movie details
@app.get("/movie/{movie_id}")
def movie_details(movie_id: int):
    if movie_id not in movie_metadata:
        raise HTTPException(404, "movie not found")
    return enrich_movie(movie_id)
