# 🌈 VibeCheck

**VibeCheck** is a lightweight emotion detection tool that reads any sentence and predicts the emotion behind it using a machine learning model trained on the [GoEmotions](https://github.com/google-research/goemotions) dataset. It gives your words a "vibe check" – are they sad? angry? joyful? neutral?

---

## 🚀 Features

- 🎯 Predicts emotions from any user-given sentence.
- 🧠 Uses a trained logistic regression classifier.
- 🔢 Supports multi-label emotion classification.
- 📦 Easily extendable to web or chatbot interfaces.

---

## 🧑‍💻 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/vibe-check.git
cd vibe-check
````

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 If `requirements.txt` isn't there, manually install:

```bash
pip install pandas scikit-learn joblib
```

### 4. Run the predictor

```bash
python3 predict.py
```

You'll be prompted to enter a sentence:

```bash
Enter a sentence: I’m feeling really good today!
Predicted Emotions: joy
```

---

## 🧪 Training the Model

If you want to retrain the model from scratch:

```bash
python3 train_model.py
```

This processes the dataset, trains the classifier, and saves the model files inside `models/`.

---

## ⚠️ Limitations

* Currently trained only on GoEmotions (28 fine-tuned emotion labels).
* Uses basic logistic regression – no deep learning or contextual embeddings yet.
* CLI-based interaction only (no web or chat UI yet).
* Struggles with sarcasm, slang, or complex sentence structure.

---

Happy VibeChecking 💫

