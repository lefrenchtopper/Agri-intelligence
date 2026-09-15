import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DB_PATH = os.path.join("data", "processed", "agri_intelligence.db")
REPORTS_DIR = os.path.join("reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

def train_forecasting_pipeline():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    print("=== Step 9: Starting Time-Series Feature Engineering & Model Training ===")
    
    df = pd.read_sql("SELECT * FROM canonical_market_prices_imputed", con=engine)
    
    # 1. Temporal Sorting & Aggregate Weekly Price across Markets
    df['month_num'] = df['month'].str.strip().str.lower().str[:3].map(MONTH_MAP)
    df['week_num'] = df['week'].fillna(1).astype(int)
    
    weekly_avg = (
        df.groupby(['year', 'month_num', 'week_num'])['modal_price_per_quintal']
        .mean()
        .reset_index()
    )
    
    # Construct continuous date index for chronological sequencing
    weekly_avg['date'] = pd.to_datetime(
        weekly_avg['year'].astype(str) + '-' + 
        weekly_avg['month_num'].astype(str).str.zfill(2) + '-01'
    )
    weekly_avg = weekly_avg.sort_values(by=['date', 'week_num']).reset_index(drop=True)

    # 2. Feature Engineering (Lags and Rolling Window Features)
    print("\nCreating temporal lag and rolling statistical features...")
    weekly_avg['lag_1'] = weekly_avg['modal_price_per_quintal'].shift(1)
    weekly_avg['lag_2'] = weekly_avg['modal_price_per_quintal'].shift(2)
    weekly_avg['lag_4'] = weekly_avg['modal_price_per_quintal'].shift(4)
    weekly_avg['rolling_mean_4'] = weekly_avg['modal_price_per_quintal'].shift(1).rolling(4).mean()
    weekly_avg['rolling_std_4'] = weekly_avg['modal_price_per_quintal'].shift(1).rolling(4).std()

    # Drop NaNs created by lag windows
    model_df = weekly_avg.dropna().reset_index(drop=True)

    features = ['year', 'month_num', 'week_num', 'lag_1', 'lag_2', 'lag_4', 'rolling_mean_4', 'rolling_std_4']
    target = 'modal_price_per_quintal'

    # 3. Chronological Train-Test Split (80% train, 20% test)
    split_idx = int(len(model_df) * 0.8)
    train = model_df.iloc[:split_idx]
    test = model_df.iloc[split_idx:].copy()

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    print(f"Training observations: {len(train)} | Test observations: {len(test)}")

    # 4. Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Predictions & Evaluation
    predictions = model.predict(X_test)
    test['predicted_price'] = predictions

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    print("\n=== Model Evaluation Metrics ===")
    print(f"Mean Absolute Error (MAE): {mae:.2f} Rs/Quintal")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Rs/Quintal")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # 6. Save Forecast Results Plot
    plt.figure(figsize=(12, 6))
    plt.plot(train['date'], train[target], label='Historical Train Prices', color='gray', alpha=0.7)
    plt.plot(test['date'], y_test, label='Actual Test Prices', color='blue', marker='o')
    plt.plot(test['date'], predictions, label='Forecasted Prices (RF)', color='orange', linestyle='--', marker='x')
    
    plt.title("Onion Market Weekly Price Forecast vs Actuals")
    plt.xlabel("Timeline")
    plt.ylabel("Price (Rs. / Quintal)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    forecast_plot_path = os.path.join(FIGURES_DIR, "price_forecast_vs_actual.png")
    plt.savefig(forecast_plot_path, dpi=300)
    plt.close()

    # Save test prediction log
    predictions_path = os.path.join(REPORTS_DIR, "weekly_forecast_eval.csv")
    test[['date', 'year', 'month_num', 'week_num', 'modal_price_per_quintal', 'predicted_price']].to_csv(predictions_path, index=False)

    print(f"\nForecast plot saved to: {forecast_plot_path}")
    print(f"Evaluation report saved to: {predictions_path}")

if __name__ == "__main__":
    train_forecasting_pipeline()