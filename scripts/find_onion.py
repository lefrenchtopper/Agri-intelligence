import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from backend.app.database import engine


with engine.connect() as connection:
    count = connection.execute(
        text("SELECT COUNT(*) FROM crops")
    ).scalar()

    print("Total crops:", count)

    result = connection.execute(
        text("""
            SELECT id, name, tamil_name, category
            FROM crops
            ORDER BY name
            LIMIT 20
        """)
    )

    for row in result:
        print(row)