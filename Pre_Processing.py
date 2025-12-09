import numpy as numpy
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Load ratings.csv
ratings = pd.read_csv("ratings.csv")

# Create rating bins for stratification
ratings['rating_bin'] = pd.cut(
    ratings['rating'],
    bins=[0, 1, 2, 3, 4, 5],
    labels=['0-1', '1-2', '2-3', '3-4', '4-5']
)

# Stratified sample of 70,000 rows
sampled_ratings, _ = train_test_split(
    ratings,
    train_size=70000,
    stratify=ratings['rating_bin'],
    random_state=None
)

# Drop rating_bin helper column
sampled_ratings = sampled_ratings.drop(columns=['rating_bin'])

# Save to new CSV
sampled_ratings.to_csv("new_sample_70k.csv", index=False)

print("Created ratings_sample_70k.csv with a balanced stratified sample.")
