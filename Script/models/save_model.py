import os
import sys
import pickle

# adding project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Script.models.recommender import model, movie_info

# Save the trained model
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("movie_info.pkl", "wb") as f:
    pickle.dump(movie_info, f)

print("Model and movie info saved successfully.")
