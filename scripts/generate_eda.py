import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

DB_PATH = os.path.join("data", "processed", "agri_intelligence.db")
REPORTS_DIR = os.path.join("reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

def generate_eda():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    print("=== Step 8: Starting Exploratory Data Analysis (EDA) ===")
    
    df = pd.read_sql("SELECT * FROM canonical_market_prices_imputed", con=engine)
    
    # 1. Market Summary Statistics
    summary_stats = df.groupby('market')['modal_price_per_quintal'].agg(
        total_records='count',
        mean_price='mean',
        min_price='min',
        max_price='max',
        std_dev='std'
    ).reset_index()
    
    summary_path = os.path.join(REPORTS_DIR, "market_summary_stats.csv")
    summary_stats.to_csv(summary_path, index=False)
    print(f"Market summary statistics saved to: {summary_path}")
    
    # Map approximate date for time-series plotting
    df['month_num'] = df['month'].str.strip().str.lower().str[:3].map(MONTH_MAP)
    df['week_num'] = df['week'].fillna(1).astype(int)
    df['date_approx'] = pd.to_datetime(
        df['year'].astype(str) + '-' + 
        df['month_num'].astype(str).str.zfill(2) + '-01'
    )
    df = df.sort_values(by=['date_approx', 'week_num'])

    # 2. Line Chart: Weekly Price Trends by Market
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='date_approx', y='modal_price_per_quintal', hue='market', marker='o')
    plt.title("Weekly Onion Modal Price Trends by Market (Coimbatore, TN)")
    plt.xlabel("Timeline")
    plt.ylabel("Price (Rs. / Quintal)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    trend_plot_path = os.path.join(FIGURES_DIR, "weekly_price_trends.png")
    plt.savefig(trend_plot_path, dpi=300)
    plt.close()
    print(f"Price trend plot saved to: {trend_plot_path}")

    # 3. Box Plot: Price Distribution across Years
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='year', y='modal_price_per_quintal', hue='year', palette='Set2', legend=False)
    plt.title("Yearly Onion Price Distribution & Volatility")
    plt.xlabel("Year")
    plt.ylabel("Price (Rs. / Quintal)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    box_plot_path = os.path.join(FIGURES_DIR, "yearly_price_distribution.png")
    plt.savefig(box_plot_path, dpi=300)
    plt.close()
    print(f"Yearly distribution plot saved to: {box_plot_path}")

    print("\n=== EDA Pipeline Completed Successfully ===")

if __name__ == "__main__":
    generate_eda()