from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_sentiment

app = FastAPI(title="AI Sentiment API")


class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "AI Sentiment API is running"
    }


@app.post("/predict")
def predict(request: TextRequest):
    result = predict_sentiment(request.text)

    return {
        "input": request.text,
        "prediction": result
    }