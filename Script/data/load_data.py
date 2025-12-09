import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


DATA_DIR = Path(os.environ.get("MOVIELENS_DATA_DIR", ".")) 

def load_ratings(sample_file: str = "sampled_data.csv"):
    """
    Load ratings dataframe from the sampled CSV file.

    """
    file_path = DATA_DIR / sample_file
    df = pd.read_csv(file_path)
    return df


def load_movies():
    """
    Load MovieLens movies.csv file (contains movieId, title, genres).
    """
    file_path = DATA_DIR / "movies.csv"
    df = pd.read_csv(file_path)
    return df

def merge_ratings_movies(ratings_df, movies_df):
    """
    Merge ratings with movies metadata on movieId.
    """
    merged = ratings_df.merge(movies_df, on="movieId", how="left")
    return merged

def load_preprocessed():
    """
    Load the fully preprocessed ratings dataset.
    """
    file_path = DATA_DIR / "data_for_recommender.csv"
    df = pd.read_csv(file_path)
    return df

def stratified_train_test_split(df, target='rating', test_size=0.2, random_state=42, n_bins=5):
    """
    Perform a stratified train/test split based on the target column.
    Useful for skewed continuous targets like MovieLens ratings.
    """
    X = df.drop(columns=[target])
    y = df[target]

    
    y_bins = pd.cut(y, bins=n_bins, labels=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y_bins
    )

    return X_train, X_test, y_train, y_test

def load_train_test(preprocessed=True, target='rating', test_size=0.2, random_state=42, n_bins=5):
    """
    Full workflow to load data and return stratified train/test splits.
    """
    if preprocessed:
        df = load_preprocessed()
    else:
        ratings = load_ratings()
        movies = load_movies()
        df = merge_ratings_movies(ratings, movies)

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        df, target=target, test_size=test_size, random_state=random_state, n_bins=n_bins
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    
    ratings = load_ratings()
    movies = load_movies()
    merged = merge_ratings_movies(ratings, movies)

    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    print("Merged shape:", merged.shape)
    print(merged.head())
