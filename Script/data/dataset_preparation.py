import pandas as pd
import numpy as np
import os


# set data paths 
DATA_DIR = os.getenv('DATA_PATH')  

if DATA_DIR is None:
    # fallback: relative path from this script
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DATA_DIR = os.path.join(BASE_DIR, "Data")

ratings_path = os.path.join(DATA_DIR, "sampled_data.csv")
movies_path = os.path.join(DATA_DIR, "movies.csv")
output_path = os.path.join(DATA_DIR, "data_for_recommender.csv")

# printing paths for verification
print("Ratings path:", ratings_path)
print("Movies path:", movies_path)
print("Output path:", output_path)

# load datasets
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

# merge ratings and movies
df = ratings.merge(movies, on="movieId", how="left")
print("Merged dataset shape:", df.shape)


# timestamp → date features

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['weekday'] = df['timestamp'].dt.weekday


# 5. multi-hot encoding for genres

df['genres'] = df['genres'].fillna("Unknown")
genre_dummies = df['genres'].str.get_dummies(sep='|')
df = pd.concat([df, genre_dummies], axis=1)

# User-level features

user_stats = df.groupby("userId")['rating'].agg(
    user_mean='mean',
    user_count='count',
    user_std='std'
).reset_index()

df = df.merge(user_stats, on="userId", how="left")
df['user_std'] = df['user_std'].fillna(0)  # for rare users

# movie-level features

movie_stats = df.groupby("movieId")['rating'].agg(
    movie_mean='mean',
    movie_count='count',
    movie_std='std'
).reset_index()

df = df.merge(movie_stats, on="movieId", how="left")
df['movie_std'] = df['movie_std'].fillna(0)  # for rare movies

# Drop unneeded columns
df = df.drop(columns=['timestamp', 'title', 'genres'])

# handle remaining missing values

df = df.fillna(0)

# shuffle data

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ===============================
# 11. Save final preprocessed dataset
# ===============================
if os.path.exists(output_path):
    print(f"Warning: {output_path} already exists and will be overwritten.")

df.to_csv(output_path, index=False)
print("Preprocessing complete.")
print(f"Saved: {output_path}")
print("Final dataset shape:", df.shape)
