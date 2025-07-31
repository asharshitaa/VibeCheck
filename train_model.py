import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

data_dir = "data/full_dataset"
all_dfs = []

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path)
        print(f"Loaded {file} with {len(df)} rows")
        all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)
print(f"Total combined rows: {len(df)}")

metadata_cols = ['id', 'author', 'subreddit', 'link_id', 'parent_id',
                 'created_utc', 'rater_id', 'example_very_unclear']
emotion_cols = [col for col in df.columns if col not in ['text'] + metadata_cols]
df = df[['text'] + emotion_cols]

def get_emotions(row):
    return [emotion for emotion in emotion_cols if row[emotion] == 1]

df['emotions'] = df.apply(get_emotions, axis=1)

df = df[df['emotions'].map(len) > 0]
print(f"✅ Filtered rows with at least one emotion: {len(df)}")

X = df['text']
y = df['emotions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)
y_test_bin = mlb.transform(y_test)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
])

pipeline.fit(X_train, y_train_bin)

y_pred = pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test_bin, y_pred, target_names=mlb.classes_))

joblib.dump(pipeline, "models/emotion_model.pkl")
joblib.dump(mlb, "models/label_binarizer.pkl")
print("✅ Model and label binarizer saved to models/")
