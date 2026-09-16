import io

import pandas as pd
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_predict_returns_price():
    payload = {
        "MedInc": 3.5,
        "HouseAge": 25,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 700,
        "AveOccup": 2.5,
        "Latitude": 36.0,
        "Longitude": -121.5,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "predicted_price" in body
    assert body["predicted_price"].startswith("$")


def test_predict_file_accepts_valid_csv():
    df = pd.DataFrame([
        {
            "MedInc": 3.5,
            "HouseAge": 25,
            "AveRooms": 5.0,
            "AveBedrms": 1.0,
            "Population": 700,
            "AveOccup": 2.5,
            "Latitude": 36.0,
            "Longitude": -121.5,
        }
    ])

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/predict-file",
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert "predicted_price_usd" in response.text
