import pandas as pd

# Load the dataset
df = pd.read_csv("dataset.csv")

# Extract emotion columns
emotion_cols = df.columns.difference(['text', 'id', 'user', 'date']) 


df[emotion_cols] = df[emotion_cols].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=emotion_cols, how='all', inplace=True)
df[emotion_cols] = df[emotion_cols].fillna(0)
df[emotion_cols] = df[emotion_cols].astype(int)


print("Cleaned dataset shape:", df.shape)
print(df.head())
