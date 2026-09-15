from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_prediction_endpoint():
    payload = {
        "year": 2024,
        "month_num": 11,
        "week_num": 2,
        "lag_1": 2400.0,
        "lag_2": 2350.0,
        "lag_4": 2200.0,
        "rolling_mean_4": 2300.0,
        "rolling_std_4": 85.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_modal_price" in data
    assert isinstance(data["predicted_modal_price"], float)
    assert data["unit"] == "Rs./Quintal"
