import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_ratings(sample: bool = True, sample_file: str = "ratings_sample_70k.csv"):
    """
    Load ratings dataframe.
    If sample=True → load the 70k sampled dataset.
    If sample=False → load full MovieLens ratings.csv (not recommended for training).
    """
    file_path = DATA_DIR / (sample_file if sample else "ratings.csv")
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
    file_path = DATA_DIR / "ratings_preprocessed.csv"
    df = pd.read_csv(file_path)
    return df


def stratified_train_test_split(df, target='rating', test_size=0.2, random_state=42, n_bins=5):
    """
    Perform a stratified train/test split based on the target column.
    Useful for skewed continuous targets like MovieLens ratings.

    Parameters:
    - df: pandas DataFrame containing features and target
    - target: name of the target column
    - test_size: proportion of the test set
    - random_state: for reproducibility
    - n_bins: number of bins for stratification (default 5 for ratings 0-5)

    Returns:
    - X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Create bins for stratification
    y_bins = pd.cut(y, bins=n_bins, labels=False)

    # Perform stratified split
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

    Parameters:
    - preprocessed: bool, if True → load preprocessed dataset, else raw merged dataset
    - target: target column name
    - test_size: proportion of test set
    - random_state: for reproducibility
    - n_bins: number of bins for stratification (default 5 for ratings 0-5)

    Returns:
    - X_train, X_test, y_train, y_test
    """
    if preprocessed:
        df = load_preprocessed()
    else:
        ratings = load_ratings()
        movies = load_movies()
        df = merge_ratings_movies(ratings, movies)
        # optionally, you could call your preprocessing function here if implemented

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        df, target=target, test_size=test_size, random_state=random_state, n_bins=n_bins
    )

    return X_train, X_test, y_train, y_test




if __name__ == "__main__":
    # Debug-run to verify everything works
    ratings = load_ratings()
    movies = load_movies()
    merged = merge_ratings_movies(ratings, movies)

    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    print("Merged shape:", merged.shape)
    print(merged.head())
