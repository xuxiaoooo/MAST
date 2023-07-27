import pandas as pd

# Load the DataFrame from the pickle file
df = pd.read_pickle('/home/user/xuxiao/MAST/data/band_csv/all.pkl').drop(columns=['id'])

# Iterate over all cells in the DataFrame
for i in df.index:
    for col in df.columns:
        if df.at[i, col].shape != (16, 2000):
            print(f"Cell at index {i}, column {col} has shape: {df.at[i, col].shape}")
