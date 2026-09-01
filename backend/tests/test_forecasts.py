import pytest
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routers import forecasts


client = TestClient(app)


@pytest.fixture
def predictions_file(tmp_path, monkeypatch):
    predictions_path = tmp_path / "adaptive_predictions.csv"
    pd.DataFrame(
        [
            {
                "district": "Coimbatore",
                "current_price": 2400.0,
                "predicted_price": 2520.0,
                "lower_price": 2250.0,
                "upper_price": 2800.0,
                "predicted_pct_change": 0.05,
                "volatility_regime": "normal",
                "spike_alert_flag": False,
            }
        ]
    ).to_csv(predictions_path, index=False)
    monkeypatch.setattr(forecasts, "PREDICTIONS_PATH", predictions_path)
    return predictions_path


def test_get_forecast_structure(predictions_file):
    response = client.get("/api/v1/forecasts/coimbatore")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["district"] == "Coimbatore"
    assert "current_price" in data
    assert "forecast" in data
    assert "predicted_price" in data["forecast"]
    assert "lower_bound" in data["forecast"]
    assert "upper_bound" in data["forecast"]
    assert "regime" in data
    assert "spike_alert" in data


def test_get_forecast_not_found(predictions_file):
    response = client.get("/api/v1/forecasts/UnknownDistrict")

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "No forecast found for district: UnknownDistrict"
    )


def test_get_forecast_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        forecasts,
        "PREDICTIONS_PATH",
        tmp_path / "missing.csv",
    )

    response = client.get("/api/v1/forecasts/Coimbatore")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Forecast predictions not generated yet."
    )
