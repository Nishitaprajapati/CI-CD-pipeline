from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "AI Sentiment API is running"


def test_predict_positive():
    response = client.post(
        "/predict",
        json={"text": "this product is amazing"}
    )

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_negative():
    response = client.post(
        "/predict",
        json={"text": "bad product"}
    )

    assert response.status_code == 200
    assert "prediction" in response.json()