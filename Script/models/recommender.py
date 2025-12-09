import os
import pandas as pd
import json
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae


# load dataset 
df_path = os.getenv('DATA_PATH')

if df_path is None:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    df_path = os.path.join(BASE_DIR, "Data", "data_for_recommender.csv")

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


# setting up an 80-20 train and test split

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)


# define the model for collaborative filtering

model = SVD(n_factors=100, n_epochs=25, random_state=42)
model.fit(train_set)


# testing the model based on the test set using RMSE and MAE metrics

predictions = model.test(test_set)

print("\nModel Evaluation:")
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
    preds = [(movie, model.predict(uid=user_id, iid=movie).est) for movie in not_watched]

    # order by predicted rating
    preds.sort(key=lambda x: x[1], reverse=True)

    # use movieId to fetch title, genres, cast
    top_n = []
    for movie, rating in preds[:n]:
        info = movie_info.get(movie, {'title': 'Unknown', 'genres': 'N/A', 'cast_names': 'N/A'})
        top_n.append((info['title'], info['genres'], info['cast_names'], rating))

    return top_n


print("\nTop recommendations for user 1:")
for title, genres, cast, rating in recommend_for_user(1, 5):
    print(f"\nTitle: {title}, Genres: {genres}, Predicted Rating: {rating:.2f}")
