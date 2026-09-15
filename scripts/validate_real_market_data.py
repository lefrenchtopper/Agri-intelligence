"""Read-only validation and provenance report for market-price source data.

This script never writes to the database, rewrites source files, generates
prices, or changes model artifacts. It reports what is present and flags
coverage, schema, date, price, and duplicate issues for later review.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    value = str(value).strip()[:10]
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def parse_price(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9.-]", "", str(value).strip())
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return reader.fieldnames or [], list(reader)


def unique_values(rows: Iterable[dict[str, str]], field: str) -> list[str]:
    return sorted({str(row.get(field, "")).strip() for row in rows if row.get(field, "").strip()})


def date_values(rows: Iterable[dict[str, str]], fields: tuple[str, ...]) -> list[date]:
    values = []
    for row in rows:
        for field in fields:
            parsed = parse_date(row.get(field))
            if parsed:
                values.append(parsed)
                break
    return values


def csv_report(path: Path, date_fields: tuple[str, ...], price_fields: tuple[str, ...], key_fields: tuple[str, ...]) -> None:
    fields, rows = read_csv(path)
    dates = date_values(rows, date_fields)
    prices = [parse_price(row.get(field)) for row in rows for field in price_fields if row.get(field)]
    prices = [value for value in prices if value is not None]
    keys = [tuple(row.get(field, "").strip() for field in key_fields) for row in rows]
    duplicates = len(keys) - len(set(keys)) if key_fields else 0

    print(f"\nDATASET: {path.relative_to(ROOT)}")
    print("  classification: source-derived or transformed source data")
    print(f"  rows: {len(rows)}")
    print(f"  columns: {fields}")
    print(f"  date range: {min(dates) if dates else 'not present'} to {max(dates) if dates else 'not present'}")
    print(f"  date fields used: {date_fields}")
    print(f"  price observations: {len(prices)}; invalid/missing price fields: {sum(1 for row in rows for field in price_fields if row.get(field) and parse_price(row.get(field)) is None)}")
    print(f"  price range: {min(prices) if prices else 'not present'} to {max(prices) if prices else 'not present'}")
    print(f"  duplicate key rows ({key_fields}): {duplicates}")

    for label, field in (("crops", "crop"), ("districts", "district"), ("markets", "market_name"), ("markets", "Market"), ("commodities", "Commodity")):
        values = unique_values(rows, field)
        if values:
            print(f"  {label} from {field}: {len(values)}; sample: {values[:8]}")


def weekly_excel_report(path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("pandas is required to inspect AGMARKNET Excel reports") from error

    frame = pd.read_excel(path, header=1)
    columns = [str(column).strip() for column in frame.columns if str(column) != "nan"]
    print(f"\nDATASET: {path.relative_to(ROOT)}")
    print("  classification: raw AGMARKNET Excel report")
    print(f"  rows below header: {len(frame)}")
    print(f"  columns: {columns}")
    print("  explicit observation date: not present in the report schema; period comes from directory path")


def report_raw_agmarknet(root: Path) -> None:
    files = sorted((root / "data" / "raw" / "agmarknet" / "2026").rglob("*.xlsx"))
    print("\nAGMARKNET 2026 EXCEL SOURCE SUMMARY")
    print(f"  report files: {len(files)}")
    periods = Counter()
    for path in files:
        parts = path.relative_to(root / "data" / "raw" / "agmarknet" / "2026").parts
        if len(parts) >= 2:
            periods[(parts[0], parts[1])] += 1
    print(f"  directory periods: {len(periods)}")
    print(f"  files per period: {sorted(set(periods.values()))}")
    if files:
        weekly_excel_report(files[0])


def report_lineage() -> None:
    print("\nPROVENANCE LINEAGE")
    print("  data/processed/agmarknet/onion_tamilnadu_weekly_2026.csv")
    print("    <- scripts/clean_agmarknet.py")
    print("    <- data/raw/agmarknet/2026/<month>/Week_<n>/*.xlsx")
    print("    transformation: reads Excel header row 1, maps four dynamic price columns,")
    print("    adds year/month/week from directory names, removes non-district rows, and")
    print("    de-duplicates by (year, month, week, district). It does not preserve week_start.")
    print("  agri_intelligence.db market_prices (10,656 rows)")
    print("    <- scripts/seed_mock_data.py")
    print("    transformation: deletes existing MarketPrice rows and generates deterministic")
    print("    prices with random.seed(42) for six crops, six districts, and synthetic markets.")
    print("  production forecast endpoint")
    print("    backend/app/routers/forecasts.py")
    print("    <- data/processed/agmarknet/model_results/adaptive_predictions.csv")
    print("  adaptive model artifact")
    print("    models/agmarknet/onion_price_adaptive_v1.joblib")
    print("    <- scripts/train_adaptive_forecaster.py")
    print("    <- data/processed/agmarknet/features/onion_tamilnadu_forecasting_features.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report real market-data provenance and quality without modifying files.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to inspect")
    args = parser.parse_args()
    root = args.root.resolve()

    print("REAL MARKET DATA VALIDATION REPORT")
    print(f"root: {root}")
    print("mode: read-only; no database, source, model, or output files are modified")

    csv_report(
        root / "data" / "raw" / "Tamilnadu.csv",
        (),
        ("Mandi WholeSale Price",),
        ("Source.Name", "Market", "Mandi WholeSale Price"),
    )
    csv_report(
        root / "data" / "raw" / "upag_onion_history.csv",
        ("record_date", "CalendarDay"),
        ("MandiWholeSalePrice",),
        ("Commodity", "District", "Market", "record_date", "MandiWholeSalePrice"),
    )
    csv_report(
        root / "data" / "processed" / "market_prices_clean.csv",
        ("price_date",),
        ("modal_price_per_quintal",),
        ("crop", "market_name", "price_date", "modal_price_per_quintal"),
    )
    csv_report(
        root / "data" / "processed" / "agmarknet" / "onion_tamilnadu_weekly_2026.csv",
        (),
        ("current_price",),
        ("year", "month", "week", "district"),
    )
    csv_report(
        root / "data" / "raw" / "agmarknet" / "maharashtra_onion_weekly.csv",
        ("price_date",),
        ("modal_price_per_quintal",),
        ("station", "price_date", "modal_price_per_quintal"),
    )
    report_raw_agmarknet(root)
    report_lineage()

    print("\nDECISION STATUS")
    print("  real 5+ year Tamil Nadu Onion coverage: not established")
    print("  synthetic database rows eligible for forecasting: no")
    print("  production model retraining authorized by this script: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())