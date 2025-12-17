import os
import pandas as pd
import json
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV, train_test_split
from surprise.accuracy import rmse, mae

# -------------------------------
# PATH LOGIC
# -------------------------------
PROJECT_ROOT = os.getenv("GITHUB_WORKSPACE") 
if not PROJECT_ROOT:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "Script", "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

df_path = os.getenv('DATA_PATH') or os.path.join(DATA_DIR, "sampled_data.csv")
mapping_path = os.path.join(DATA_DIR, "movie_mapping.csv")

print(f"DEBUG: Project Root: {PROJECT_ROOT}")

# -------------------------------
# Load datasets
# -------------------------------
df = pd.read_csv(df_path)
mapping_df = pd.read_csv(mapping_path)

# -------------------------------
# FORCE COLUMN ALIGNMENT (Fixes KeyErrors)
# -------------------------------
# 1. Strip whitespace from headers
mapping_df.columns = [str(c).strip() for c in mapping_df.columns]
print(f"DEBUG: Original Columns: {list(mapping_df.columns)}")

# 2. Map existing columns to standard names used by the script
col_map = {}
for c in mapping_df.columns:
    low_c = c.lower()
    if low_c in ['movieid', 'id']: col_map[c] = 'movieId'
    if low_c in ['genres', 'genre']: col_map[c] = 'genres'
    if low_c in ['title', 'name']: col_map[c] = 'title'
    if low_c in ['cast']: col_map[c] = 'cast'

mapping_df = mapping_df.rename(columns=col_map)

# 3. Final Fallback: If 'movieId' is still missing, take the first column
if 'movieId' not in mapping_df.columns:
    print("WARNING: Could not find movieId column by name. Using first column as ID.")
    mapping_df = mapping_df.rename(columns={mapping_df.columns[0]: 'movieId'})

# 4. Fill missing columns with N/A to prevent crashes
for req in ['genres', 'cast', 'title']:
    if req not in mapping_df.columns:
        print(f"WARNING: '{req}' column missing. Creating dummy column.")
        mapping_df[req] = "N/A"

mapping_df['genres'] = mapping_df['genres'].fillna("N/A")
mapping_df['cast'] = mapping_df['cast'].fillna("[]")

# -------------------------------
# Process Movie Metadata
# -------------------------------
def extract_cast_names(cast_str, top_n=5):
    try:
        if isinstance(cast_str, list):
            cast_list = cast_str
        else:
            cast_list = json.loads(cast_str)
        names = [c.get('name', 'Unknown') for c in cast_list]
        return ', '.join(names[:top_n])
    except Exception:
        return "N/A"

mapping_df['cast_names'] = mapping_df['cast'].apply(extract_cast_names)

# Create movie_info dictionary - Now guaranteed to have 'movieId'
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

param_grid = {
    'n_factors': [50, 100],
    'n_epochs': [20, 50],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.05]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

best_model = SVD(**gs.best_params['rmse'])
trainset = data.build_full_trainset()
best_model.fit(trainset)

# -------------------------------
# Save Artifacts
# -------------------------------
def save_pickle(obj, filename):
    path = os.path.join(SAVED_MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

save_pickle(best_model, "trained_collaborative_model.pkl")
save_pickle(movie_info, "hybrid_movie_metadata.pkl")

print(f"SUCCESS: Collaborative artifacts saved in {SAVED_MODELS_DIR}")

if __name__ == "__main__":
    all_movies = df['movieId'].unique()
    preds = [(m, best_model.predict(uid=1, iid=m).est) for m in all_movies[:5]]
    print(f"\nSample Predictions for User 1:")
    for mid, score in preds:
        title = movie_info.get(int(mid), {}).get('title', 'Unknown')
        print(f"- {title}: {score:.2f}")