import joblib
import os

MODEL_PATH = "sentiment_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Run train_model.py first.")

    return joblib.load(MODEL_PATH)


model = load_model()


def predict_sentiment(text: str):
    prediction = model.predict([text])
    return prediction[0]