import os
import pickle
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel

# --- SMART PATH LOGIC ---
# Get the absolute path of the directory where backend.py is located (Script/fastapi)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Move up one level to the 'Script' folder
SCRIPT_DIR = os.path.dirname(CURRENT_DIR)
# Move up again to get the Project Root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Define specific folder paths
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_models")
FRONTEND_DIR = os.path.join(SCRIPT_DIR, "frontend") # Path to Script/frontend

# Diagnostics printed to Hugging Face logs
print(f"--- STARTUP DIAGNOSTICS ---")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"FRONTEND_DIR: {FRONTEND_DIR}")
print(f"FRONTEND_EXISTS: {os.path.exists(FRONTEND_DIR)}")
if os.path.exists(FRONTEND_DIR):
    print(f"FRONTEND_CONTENTS: {os.listdir(FRONTEND_DIR)}")
print(f"---------------------------")

# Initialize globals
similarity_matrix = None
movie_index_map = None
movie_metadata = {}
collaborative_model = None
ALL_MOVIES = []

# Load dataframes once at startup
try:
    csv_path = os.path.join(DATA_DIR, "sampled_data.csv")
    sampled_df = pd.read_csv(csv_path)
except Exception as e:
    print(f"CRITICAL: Could not load CSV data at {csv_path}: {e}")

USERS = {
    "abdullah": {"user_id": 1, "password": "1234", "role": "user"},
    "admin": {"user_id": 2, "password": "admin", "role": "admin"}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global similarity_matrix, movie_index_map, movie_metadata, collaborative_model, ALL_MOVIES
    
    def load_pickle(name):
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"ERROR: {name}: {e}")
        return None

    collaborative_model = load_pickle("hybrid_cf_model.pkl") or load_pickle("trained_collaborative_model.pkl")
    similarity_matrix = load_pickle("hybrid_similarity_matrix.pkl")
    movie_index_map = load_pickle("hybrid_movie_index_map.pkl")
    movie_metadata = load_pickle("hybrid_movie_metadata.pkl") or load_pickle("movie_metadata.pkl") or {}

    if movie_index_map:
        ALL_MOVIES = list(movie_index_map.keys())
    
    yield

app = FastAPI(
    title="Cinephile API",
    lifespan=lifespan
)

# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FRONTEND ROUTING (FIXED) ---

# 1. Serve the index.html at the ROOT URL "/"
@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "error": "Frontend index.html not found",
        "checked_path": index_file,
        "hint": "Check if your folder is named 'frontend' inside 'Script'"
    }

# 2. Mount the Script/frontend folder so the HTML can find /static/style.css etc.
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# --- TMDB & LOGIC ---
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def tmdb_search_movie(title: str):
    if not TMDB_API_KEY: return {}
    try:
        r = requests.get(f"{TMDB_BASE_URL}/search/movie", params={"api_key": TMDB_API_KEY, "query": title}, timeout=5)
        return r.json()["results"][0] if r.status_code == 200 and r.json().get("results") else {}
    except: return {}

def enrich_movie(movie_id: int):
    meta = movie_metadata.get(movie_id, {})
    title = meta.get("title", "Unknown")
    tmdb_data = tmdb_search_movie(title)
    return {
        "movie_id": int(movie_id),
        "title": title,
        "genres": meta.get("genres", "N/A"),
        "cast": meta.get("cast_names", "N/A"),
        "overview": tmdb_data.get("overview"),
        "poster": f"https://image.tmdb.org/t/p/w500{tmdb_data['poster_path']}" if tmdb_data.get("poster_path") else None,
        "backdrop": f"https://image.tmdb.org/t/p/w780{tmdb_data['backdrop_path']}" if tmdb_data.get("backdrop_path") else None,
        "release_date": tmdb_data.get("release_date"),
        "rating_tmdb": tmdb_data.get("vote_average")
    }

def content_score(user_id: int, movie_id: int):
    if movie_id not in movie_index_map or similarity_matrix is None:
        return 2.75
    target_idx = movie_index_map[movie_id]
    try:
        user_ratings = collaborative_model.trainset.ur.get(user_id, [])
    except: return 2.75
    if not user_ratings: return 2.75
    score_sum, weight_sum = 0.0, 0.0
    for inner_iid, rating in user_ratings:
        try:
            raw_id = int(collaborative_model.trainset.to_raw_iid(inner_iid))
            if raw_id in movie_index_map:
                idx = movie_index_map[raw_id]
                sim = similarity_matrix[target_idx][idx]
                score_sum += sim * rating
                weight_sum += abs(sim)
        except: continue
    if weight_sum == 0: return 2.75
    return 0.5 + 4.5 * np.clip(score_sum / weight_sum, 0.0, 1.0)

def hybrid_predict(user_id: int, movie_id: int, alpha: float):
    cf = collaborative_model.predict(user_id, movie_id).est
    cb = content_score(user_id, movie_id)
    return alpha * cf + (1 - alpha) * cb

# --- API ENDPOINTS ---
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": collaborative_model is not None}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(data: LoginRequest):
    user = USERS.get(data.username)
    if not user or user["password"] != data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user_id": user["user_id"], "role": user["role"], "username": data.username}

@app.get("/recommend")
def recommend(user_id: int, n: int = Query(10, le=50), alpha: float = Query(0.7, ge=0.0, le=1.0)):
    if collaborative_model is None or not ALL_MOVIES:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        inner_uid = collaborative_model.trainset.to_inner_uid(int(user_id))
        watched = {int(collaborative_model.trainset.to_raw_iid(iid)) for iid, _ in collaborative_model.trainset.ur.get(inner_uid, [])}
    except: return []
    candidates = [m for m in ALL_MOVIES if m not in watched]
    scores = []
    for m in candidates:
        try: scores.append((m, hybrid_predict(user_id, m, alpha)))
        except: continue
    scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for mid, score in scores[:n]:
        data = enrich_movie(mid)
        data["predicted_rating"] = round(float(score), 3)
        results.append(data)
    return results

@app.get("/user/history")
def user_history(user_id: int):
    try:
        inner_uid = collaborative_model.trainset.to_inner_uid(int(user_id))
        user_ratings = collaborative_model.trainset.ur.get(inner_uid, [])
        history = []
        for inner_iid, rating in user_ratings:
            mid = int(collaborative_model.trainset.to_raw_iid(inner_iid))
            movie_data = enrich_movie(mid)
            history.append({"movie_id": mid, "title": movie_data.get("title"), "poster": movie_data.get("poster"), "rating": float(rating)})
        return history
    except: return []

@app.get("/search")
def search_movies(query: str = Query(..., min_length=1)):
    results = []
    for mid, meta in movie_metadata.items():
        if query.lower() in meta.get("title", "").lower():
            results.append(enrich_movie(mid))
    return results[:20]

@app.get("/admin/stats")
def admin_stats(username: str = None):
    if username != "admin": raise HTTPException(status_code=403, detail="Unauthorized")
    return {"total_users": len(USERS), "total_movies": len(ALL_MOVIES)}