import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import engine
from backend.app.models import Crop


CROPS = [
    {
        "name": "Onion",
        "tamil_name": "வெங்காயம்",
        "category": "Vegetable",
        "growing_days_min": 90,
        "growing_days_max": 150,
        "min_temp_c": 13.0,
        "max_temp_c": 25.0,
        "min_rainfall_mm": 350.0,
        "max_rainfall_mm": 550.0,
        "water_requirement_mm": 350.0,
        "soil_compatibility": ["Loamy", "Sandy Loam", "Alluvial"],
    },
    {
        "name": "Tomato",
        "tamil_name": "தக்காளி",
        "category": "Vegetable",
        "growing_days_min": 90,
        "growing_days_max": 150,
        "min_temp_c": 18.0,
        "max_temp_c": 30.0,
        "min_rainfall_mm": 400.0,
        "max_rainfall_mm": 600.0,
        "water_requirement_mm": 400.0,
        "soil_compatibility": ["Loamy", "Sandy Loam", "Red Soil"],
    },
    {
        "name": "Coconut",
        "tamil_name": "தேங்காய்",
        "category": "Plantation",
        "growing_days_min": 365,
        "growing_days_max": 3650,
        "min_temp_c": 21.0,
        "max_temp_c": 32.0,
        "min_rainfall_mm": 1000.0,
        "max_rainfall_mm": 2500.0,
        "water_requirement_mm": 1500.0,
        "soil_compatibility": ["Sandy Loam", "Loamy", "Alluvial"],
    },
    {
        "name": "Rice",
        "tamil_name": "நெல்",
        "category": "Cereal",
        "growing_days_min": 110,
        "growing_days_max": 150,
        "min_temp_c": 20.0,
        "max_temp_c": 35.0,
        "min_rainfall_mm": 1000.0,
        "max_rainfall_mm": 2000.0,
        "water_requirement_mm": 1200.0,
        "soil_compatibility": ["Clay", "Clay Loam", "Alluvial"],
    },
    {
        "name": "Maize",
        "tamil_name": "மக்காச்சோளம்",
        "category": "Cereal",
        "growing_days_min": 90,
        "growing_days_max": 120,
        "min_temp_c": 18.0,
        "max_temp_c": 32.0,
        "min_rainfall_mm": 500.0,
        "max_rainfall_mm": 800.0,
        "water_requirement_mm": 500.0,
        "soil_compatibility": ["Loamy", "Sandy Loam", "Alluvial"],
    },
    {
        "name": "Groundnut",
        "tamil_name": "நிலக்கடலை",
        "category": "Oilseed",
        "growing_days_min": 100,
        "growing_days_max": 140,
        "min_temp_c": 20.0,
        "max_temp_c": 30.0,
        "min_rainfall_mm": 500.0,
        "max_rainfall_mm": 1000.0,
        "water_requirement_mm": 500.0,
        "soil_compatibility": ["Sandy Loam", "Red Soil", "Loamy"],
    },
]


with Session(engine) as session:
    added = 0

    for crop_data in CROPS:
        existing = session.scalar(
            select(Crop).where(Crop.name == crop_data["name"])
        )

        if existing:
            print(f"Already exists: {crop_data['name']}")
            continue

        session.add(Crop(**crop_data))
        added += 1
        print(f"Added: {crop_data['name']}")

    session.commit()

    print(f"\nDone. Added {added} crops.")