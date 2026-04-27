from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

texts = [
    "this product is good",
    "i love this service",
    "this is amazing",
    "very happy with the result",
    "bad product",
    "i hate this service",
    "very poor experience",
    "this is terrible",
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative",
]

model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", LogisticRegression())
])

model.fit(texts, labels)

joblib.dump(model, "sentiment_model.pkl")

print("Model trained and saved successfully.")