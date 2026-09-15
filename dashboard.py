import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Agri-Market Intelligence Dashboard",
    page_icon="🌾",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("🌾 Agri-Market Intelligence Price Forecasting")
st.markdown("Real-time weekly modal price predictions for Coimbatore market commodities.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Parameters")
    year = st.number_input("Year", min_value=2020, max_value=2030, value=2024)
    month_num = st.slider("Month", min_value=1, max_value=12, value=11)
    week_num = st.slider("Week of Month", min_value=1, max_value=5, value=2)
    
    st.markdown("---")
    st.markdown("**Historical Feature Inputs (Rs. / Quintal)**")
    lag_1 = st.number_input("1-Week Lag Price (Lag 1)", value=2400.0)
    lag_2 = st.number_input("2-Week Lag Price (Lag 2)", value=2350.0)
    lag_4 = st.number_input("4-Week Lag Price (Lag 4)", value=2200.0)
    rolling_mean_4 = st.number_input("4-Week Rolling Mean", value=2300.0)
    rolling_std_4 = st.number_input("4-Week Rolling Std Dev", value=85.5)

with col2:
    st.subheader("Price Prediction Output")
    
    if st.button("Generate Forecast", type="primary"):
        payload = {
            "year": int(year),
            "month_num": int(month_num),
            "week_num": int(week_num),
            "lag_1": float(lag_1),
            "lag_2": float(lag_2),
            "lag_4": float(lag_4),
            "rolling_mean_4": float(rolling_mean_4),
            "rolling_std_4": float(rolling_std_4)
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                predicted_price = result["predicted_modal_price"]
                unit = result["unit"]
                
                st.metric(
                    label="Predicted Modal Price",
                    value=f"₹ {predicted_price:,.2f}",
                    delta=f"{predicted_price - lag_1:+.2f} vs last week"
                )
                st.success(f"Successfully generated prediction via FastAPI backend.")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Could not connect to FastAPI server at {API_URL}. Ensure uvicorn is running.")