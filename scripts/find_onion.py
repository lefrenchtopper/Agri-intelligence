import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from backend.app.database import engine

with engine.connect() as connection:
    query = text("""
        SELECT id, name, tamil_name, category
        FROM crops
        WHERE LOWER(name) LIKE :search_term
           OR LOWER(tamil_name) LIKE :search_term
        ORDER BY id
    """)

    result = connection.execute(query, {"search_term": "%onion%"})
    rows = result.fetchall()

    print(f"Found {len(rows)} matching crop record(s) for 'onion':\n")
    for row in rows:
        print(f"ID: {row.id} | Name: {row.name} | Tamil Name: {row.tamil_name} | Category: {row.category}")