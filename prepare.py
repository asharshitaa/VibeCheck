import pandas as pd
import os

data_dir = 'data/full_dataset'
files = ['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv']

df = pd.concat([pd.read_csv(os.path.join(data_dir, f), names=["text", "labels", "id"]) for f in files], ignore_index=True)
df.dropna(inplace=True)

print("Sample rows:")
print(df.head())
print(f"\nTotal records: {len(df)}")
