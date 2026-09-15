import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor

DB_PATH = os.path.join("data", "processed", "agri_intelligence.db")
MODEL_PATH = os.path.join("app", "model.joblib")

os.makedirs("app", exist_ok=True)

MONTH_MAP = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

engine = create_engine(f"sqlite:///{DB_PATH}")
df = pd.read_sql("SELECT * FROM canonical_market_prices_imputed", con=engine)

df['month_num'] = df['month'].str.strip().str.lower().str[:3].map(MONTH_MAP)
df['week_num'] = df['week'].fillna(1).astype(int)

weekly_avg = df.groupby(['year', 'month_num', 'week_num'])['modal_price_per_quintal'].mean().reset_index()
weekly_avg = weekly_avg.sort_values(by=['year', 'month_num', 'week_num']).reset_index(drop=True)

weekly_avg['lag_1'] = weekly_avg['modal_price_per_quintal'].shift(1)
weekly_avg['lag_2'] = weekly_avg['modal_price_per_quintal'].shift(2)
weekly_avg['lag_4'] = weekly_avg['modal_price_per_quintal'].shift(4)
weekly_avg['rolling_mean_4'] = weekly_avg['modal_price_per_quintal'].shift(1).rolling(4).mean()
weekly_avg['rolling_std_4'] = weekly_avg['modal_price_per_quintal'].shift(1).rolling(4).std()

model_df = weekly_avg.dropna()
features = ['year', 'month_num', 'week_num', 'lag_1', 'lag_2', 'lag_4', 'rolling_mean_4', 'rolling_std_4']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(model_df[features], model_df['modal_price_per_quintal'])

joblib.dump(model, MODEL_PATH)
print(f"Model exported successfully to {MODEL_PATH}")