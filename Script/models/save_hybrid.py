import os
import sys
import pickle

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import hybrid components
from Script.models.hybrid import (
    similarity_matrix,
    movie_index_map,
    movie_metadata,
    collaborative_model
)

# content-based similarity matrix
with open("hybrid_similarity_matrix.pkl", "wb") as f:
    pickle.dump(similarity_matrix, f)

# movie index map (movieId -> row index in similarity matrix)
with open("hybrid_movie_index_map.pkl", "wb") as f:
    pickle.dump(movie_index_map, f)

# movie metadata (title, genres, cast)
with open("hybrid_movie_metadata.pkl", "wb") as f:
    pickle.dump(movie_metadata, f)

# collaborative filtering model used in hybrid
with open("hybrid_cf_model.pkl", "wb") as f:
    pickle.dump(collaborative_model, f)

print("Hybrid model artifacts saved successfully.")
