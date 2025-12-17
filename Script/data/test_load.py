# test_load.py
from load_data import load_preprocessed

# Load the preprocessed dataset
df = load_preprocessed()

# Check its shape and first few rows
print("Dataset shape:", df.shape)
print(df.head())
