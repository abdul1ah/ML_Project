import os
import sys
import pickle

# adding project root 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Script.models.collaborative import best_model, movie_info

# Save the trained model
with open("trained_collaborative_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("movie_info.pkl", "wb") as f:
    pickle.dump(movie_info, f)

print("Model and movie info saved successfully.")
