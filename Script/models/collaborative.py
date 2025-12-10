import os
import pandas as pd
import json
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV, train_test_split
from surprise.accuracy import rmse, mae

# load dataset 
df_path = os.getenv('DATA_PATH')

if df_path is None:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    df_path = os.path.join(BASE_DIR, "Data", "sampled_data.csv")   

df = pd.read_csv(df_path)
print("Dataset shape:", df.shape)

# load movies mapping file (title, genres, cast)
mapping_df = pd.read_csv(os.path.join(BASE_DIR, "Data", "movie_mapping.csv"))

# using a dictionary for easy lookup
# fill N/A to avoid errors with missing genres or cast
mapping_df['genres'] = mapping_df['genres'].fillna("N/A")
mapping_df['cast'] = mapping_df['cast'].fillna("[]")  # empty JSON list if missing

# extract top 5 actor names from cast JSON string
def extract_cast_names(cast_str, top_n=5):
    try:
        cast_list = json.loads(cast_str)
        names = [c['name'] for c in cast_list]
        return ', '.join(names[:top_n])
    except Exception:
        return "N/A"

mapping_df['cast_names'] = mapping_df['cast'].apply(extract_cast_names)
movie_info = mapping_df.set_index('movieId')[['title', 'genres', 'cast_names']].to_dict(orient='index')

# restructuring into surprise's required format
reader = Reader(rating_scale=(df.rating.min(), df.rating.max()))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# perform grid search to find best hyperparameters (smaller grid for lighter computation)
param_grid = {
    'n_factors': [50, 100],
    'n_epochs': [20, 50],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.05]
}

# GridSearchCV with 3-fold cross-validation (simpler, lighter)
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)  # removed joblib_verbose
gs.fit(data)

print("\nBest RMSE:", gs.best_score['rmse'])
print("Best MAE:", gs.best_score['mae'])
print("Best parameters:", gs.best_params['rmse'])

# train the model with best parameters on the full dataset
best_model = SVD(**gs.best_params['rmse'])
trainset = data.build_full_trainset()
best_model.fit(trainset)

# optional 80-20 test split for evaluation
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# testing the model based on the test set using RMSE and MAE metrics
predictions = best_model.test(test_set)
print("\nModel Evaluation on test set:")
rmse(predictions)
mae(predictions)

# recommend top 5 movies for a user
def recommend_for_user(user_id, n=5):
    
    all_movies = df['movieId'].unique()
    
    # movies already watched
    watched = df[df['userId'] == user_id]['movieId'].values

    # movies not yet watched
    not_watched = [m for m in all_movies if m not in watched]

    # predict ratings
    preds = [(movie, best_model.predict(uid=user_id, iid=movie).est) for movie in not_watched]

    # order by predicted rating
    preds.sort(key=lambda x: x[1], reverse=True)

    # use movieId to fetch title, genres, cast
    top_n = []
    for movie, rating in preds[:n]:
        info = movie_info.get(movie, {'title': 'Unknown', 'genres': 'N/A', 'cast_names': 'N/A'})
        top_n.append((info['title'], info['genres'], info['cast_names'], rating))

    return top_n

# test recommendations for a sample user
print("\nTop recommendations for user 1:")
for title, genres, cast, rating in recommend_for_user(1, 5):
    print(f"\nTitle: {title}, Genres: {genres}, Predicted Rating: {rating:.2f}")
