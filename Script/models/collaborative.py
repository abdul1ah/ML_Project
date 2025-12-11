import os
import pandas as pd
import json
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV, train_test_split
from surprise.accuracy import rmse, mae

# -------------------------------
# Load ratings dataset
# -------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
df_path = os.getenv('DATA_PATH') or os.path.join(BASE_DIR, "Data", "sampled_data.csv")
df = pd.read_csv(df_path)
print("Dataset shape:", df.shape)

# -------------------------------
# Load movie mapping (title, genres, cast)
# -------------------------------
mapping_df = pd.read_csv(os.path.join(BASE_DIR, "Data", "movie_mapping.csv"))
mapping_df['genres'] = mapping_df['genres'].fillna("N/A")
mapping_df['cast'] = mapping_df['cast'].fillna("[]")

# Extract top 5 actor names
def extract_cast_names(cast_str, top_n=5):
    try:
        cast_list = json.loads(cast_str)
        names = [c['name'] for c in cast_list]
        return ', '.join(names[:top_n])
    except Exception:
        return "N/A"

mapping_df['cast_names'] = mapping_df['cast'].apply(extract_cast_names)

# Create movie_info dictionary
movie_info = mapping_df.set_index('movieId')[['title', 'genres', 'cast_names']].to_dict(orient='index')

# Ensure all movies in df are in movie_info
for mid in df['movieId'].unique():
    if mid not in movie_info:
        movie_info[mid] = {'title': 'Unknown', 'genres': 'N/A', 'cast_names': 'N/A'}

# -------------------------------
# Prepare Surprise dataset
# -------------------------------
reader = Reader(rating_scale=(df.rating.min(), df.rating.max()))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# -------------------------------
# GridSearchCV for best hyperparameters
# -------------------------------
param_grid = {
    'n_factors': [50, 100],
    'n_epochs': [20, 50],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.05]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

print("\nBest RMSE:", gs.best_score['rmse'])
print("Best MAE:", gs.best_score['mae'])
print("Best parameters:", gs.best_params['rmse'])

# -------------------------------
# Train best model on full dataset
# -------------------------------
best_model = SVD(**gs.best_params['rmse'])
trainset = data.build_full_trainset()
best_model.fit(trainset)

# Optional evaluation on 20% test split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
predictions = best_model.test(test_set)
print("\nModel Evaluation on test set:")
rmse(predictions)
mae(predictions)

# -------------------------------
# Function to recommend top N movies for a user
# -------------------------------
def recommend_for_user(user_id, n=5):
    all_movies = df['movieId'].unique()
    watched = df[df['userId'] == user_id]['movieId'].values
    not_watched = [m for m in all_movies if m not in watched]

    preds = [(movie, best_model.predict(uid=user_id, iid=movie).est) for movie in not_watched]
    preds.sort(key=lambda x: x[1], reverse=True)

    top_n = []
    for movie, rating in preds[:n]:
        info = movie_info.get(movie, {'title': 'Unknown', 'genres': 'N/A', 'cast_names': 'N/A'})
        top_n.append((info['title'], info['genres'], info['cast_names'], rating))
    return top_n

print("\nTop recommendations for user 1:")
for title, genres, cast, rating in recommend_for_user(1, 5):
    print(f"\nTitle: {title}, Genres: {genres}, Cast: {cast}, Predicted Rating: {rating:.2f}")

# -------------------------------
# Save trained model and movie info
# -------------------------------
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "Script", "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

with open(os.path.join(SAVED_MODELS_DIR, "trained_collaborative_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

with open(os.path.join(SAVED_MODELS_DIR, "movie_info.pkl"), "wb") as f:
    pickle.dump(movie_info, f)

print("\nTrained collaborative model and movie info saved successfully.")
