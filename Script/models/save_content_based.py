import os
import sys
import pickle

# adding project root 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Script.models.content_based import similarity_matrix, movie_index, ratings_df

# save the precomputed similarity matrix
with open("movie_similarity_matrix.pkl", "wb") as f:
    pickle.dump(similarity_matrix, f)

# save the movie index mapping (movieId -> row index in similarity matrix)
with open("content_movie_index.pkl", "wb") as f:
    pickle.dump(movie_index, f)

# save user ratings (for fallback predictions or updates)
with open("content_ratings_df.pkl", "wb") as f:
    pickle.dump(ratings_df, f)

print("Content-based model artifacts saved successfully.")
