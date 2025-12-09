import os
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae

# ===============================
# 1. Load dataset (container-ready)
# ===============================
# Check for environment variable first
df_path = os.getenv('DATA_PATH')

if df_path is None:
    # fallback: relative path from this script
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    df_path = os.path.join(BASE_DIR, "Data", "ratings_preprocessed.csv")

df = pd.read_csv(df_path)

# Surprise requires a specific format
reader = Reader(rating_scale=(df.rating.min(), df.rating.max()))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# ===============================
# 2. Train/test split
# ===============================
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# ===============================
# 3. Build the model (SVD CF)
# ===============================
model = SVD(n_factors=100, n_epochs=25, random_state=42)
model.fit(train_set)

# ===============================
# 4. Evaluate
# ===============================
predictions = model.test(test_set)

print("\nModel Evaluation:")
rmse(predictions)
mae(predictions)

# ===============================
# 5. Recommend top N movies for a user
# ===============================
def recommend_for_user(user_id, n=10):
    # all movie IDs
    all_movies = df['movieId'].unique()

    # movies already watched
    watched = df[df['userId'] == user_id]['movieId'].values

    # movies not yet watched
    not_watched = [m for m in all_movies if m not in watched]

    # predict ratings
    preds = [(movie, model.predict(uid=user_id, iid=movie).est) for movie in not_watched]

    # sort by predicted rating
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:n]

# Example usage:
print("\nTop recommendations for user 1:")
print(recommend_for_user(1, 10))
