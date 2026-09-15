import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from backend.app.database import engine

with engine.connect() as connection:
    result = connection.execute(
        text("""
            SELECT 
                mp.id,
                c.name AS crop_name,
                mp.market_name,
                mp.district,
                mp.price_date,
                mp.modal_price_per_quintal,
                mp.arrival_quantity_quintals
            FROM market_prices mp
            JOIN crops c ON mp.crop_id = c.id
            ORDER BY mp.created_at DESC
            LIMIT 10
        """)
    )

    print("📊 Current Market Price Records in DB:\n")
    for row in result:
        print(f"Crop: {row.crop_name} | Market: {row.market_name} ({row.district}) | Date: {row.price_date} | Modal Price: ₹{row.modal_price_per_quintal}/qtl | Arrival: {row.arrival_quantity_quintals} qtl")