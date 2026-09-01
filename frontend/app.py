from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv(
    "AGRI_API_BASE_URL",
    "http://localhost:8000/api/v1/forecasts",
)
DISTRICTS = [
    "Coimbatore",
    "Chennai",
    "Madurai",
    "Kallakurichi",
    "Tiruppur",
    "Tiruchirappalli",
]
REQUEST_TIMEOUT_SECONDS = 10


st.set_page_config(
    page_title="Agri-Intelligence Forecasts",
    page_icon="AI",
    layout="wide",
)

st.title("Agricultural Price Forecasts")
st.caption("Weekly onion prices, predictive quantile bounds, and market regime alerts")

with st.sidebar:
    st.header("Forecast lookup")
    district = st.selectbox("Target district", DISTRICTS)
    fetch_forecast = st.button(
        "Fetch forecast",
        type="primary",
        use_container_width=True,
    )
    st.caption(f"API: {API_BASE_URL}")


def fetch_district_forecast(selected_district: str) -> dict | None:
    try:
        response = requests.get(
            f"{API_BASE_URL}/{selected_district}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError:
        st.error(
            "The forecast API is unavailable. Start it with "
            "`uvicorn backend.app.main:app --reload`."
        )
        return None
    except requests.exceptions.RequestException as error:
        st.error(f"Could not fetch the forecast: {error}")
        return None

    if response.status_code == 404:
        st.warning(f"No forecast record was found for {selected_district}.")
        return None

    if response.status_code == 503:
        st.error(
            "Forecast data has not been generated yet. Run "
            "`predict_adaptive_forecaster.py` first."
        )
        return None

    if response.status_code != 200:
        st.error(f"The forecast API returned HTTP {response.status_code}.")
        return None

    try:
        return response.json()
    except ValueError:
        st.error("The forecast API returned an invalid response.")
        return None


if fetch_forecast or "forecast_data" not in st.session_state:
    with st.spinner(f"Loading the latest {district} forecast..."):
        st.session_state.forecast_data = fetch_district_forecast(district)

forecast_data = st.session_state.get("forecast_data")

if forecast_data:
    forecast = forecast_data["forecast"]
    predicted_change = float(forecast["predicted_pct_change"])
    is_alert = bool(forecast_data["spike_alert"])

    if is_alert:
        st.error(
            f"HIGH VOLATILITY ALERT | {forecast_data['district']} | "
            f"Market regime: {forecast_data['regime'].upper()}"
        )
    else:
        st.success(
            f"STABLE MARKET | {forecast_data['district']} | "
            f"Market regime: {forecast_data['regime'].upper()}"
        )

    current_price = float(forecast_data["current_price"])
    predicted_price = float(forecast["predicted_price"])
    lower_bound = float(forecast["lower_bound"])
    upper_bound = float(forecast["upper_bound"])

    current_col, predicted_col, lower_col, upper_col = st.columns(4)
    current_col.metric("Current price", f"Rs {current_price:,.2f}")
    predicted_col.metric(
        "Predicted price",
        f"Rs {predicted_price:,.2f}",
        delta=f"{predicted_change * 100:+.2f}%",
    )
    lower_col.metric("Lower bound (10%)", f"Rs {lower_bound:,.2f}")
    upper_col.metric("Upper bound (90%)", f"Rs {upper_bound:,.2f}")

    st.subheader("Forecast interval")
    chart_data = pd.DataFrame(
        {
            "Price": [lower_bound, predicted_price, upper_bound],
        },
        index=["Lower bound", "Predicted", "Upper bound"],
    )
    st.bar_chart(chart_data, y="Price", height=320)

    details_col, interpretation_col = st.columns(2)
    with details_col:
        st.subheader("Forecast details")
        st.dataframe(
            pd.DataFrame(
                {
                    "Measure": [
                        "District",
                        "Market regime",
                        "Predicted change",
                        "Alert status",
                    ],
                    "Value": [
                        forecast_data["district"],
                        forecast_data["regime"].title(),
                        f"{predicted_change * 100:+.2f}%",
                        "Alert" if is_alert else "Normal",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with interpretation_col:
        st.subheader("Decision signal")
        if is_alert:
            st.write(
                "The model identifies elevated volatility or a projected "
                "price increase above the alert threshold. Review supply "
                "conditions before making procurement decisions."
            )
        else:
            st.write(
                "The model does not identify an immediate surge signal. "
                "Continue monitoring the prediction interval and incoming "
                "market prices."
            )
