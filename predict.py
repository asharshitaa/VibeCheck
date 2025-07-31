import joblib

# Load model and label binarizer
model = joblib.load("models/emotion_model.pkl")
mlb = joblib.load("models/label_binarizer.pkl")

# Accept user input
text = input("Enter a sentence: ")

# Predict
pred = model.predict([text])
labels = mlb.inverse_transform(pred)

# Output
if labels[0]:
    print("\nPredicted Emotions:", ", ".join(labels[0]))
else:
    print("\nPredicted Emotions: neutral")
