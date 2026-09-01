import argparse
import io
from pathlib import Path

import pandas as pd
import requests


ENDPOINT = "https://api.agmarknet.gov.in/v1/price-trend/wholesale-prices-weekly"
DEFAULT_OUTPUT = Path("data/raw/agmarknet/maharashtra_onion_weekly.csv")
DEFAULT_RAW_DIR = Path("data/raw/agmarknet/maharashtra_weekly")
STATION_PATTERN = r"nashik|nasik"


def fetch_report(session, year, month, week, state_id):
    response = session.get(
        ENDPOINT,
        params={
            "report_mode": "Districtwise",
            "commodity": 23,
            "year": year,
            "month": month,
            "week": week,
            "state": state_id,
            "district": 0,
            "export": "true",
        },
        headers={"Accept": "*/*", "User-Agent": "Mozilla/5.0"},
        timeout=45,
    )
    response.raise_for_status()
    if not response.content.startswith(b"PK"):
        raise ValueError(f"Unexpected AGMARKNET response for {year}-{month}-{week}")
    return response.content


def normalize_report(content, year, month, week):
    frame = pd.read_excel(io.BytesIO(content), header=1)
    normalized = {
        str(column).strip().casefold(): column for column in frame.columns
    }
    market_column = next(
        (
            normalized[key]
            for key in normalized
            if "district" in key or "market" in key or "mandi" in key
        ),
        None,
    )
    price_column = next(
        (
            normalized[key]
            for key in normalized
            if "prices" in key or "modal" in key or "wholesale" in key or "price" in key
        ),
        None,
    )
    if market_column is None or price_column is None:
        return pd.DataFrame()

    station_text = frame[market_column].astype(str)
    selected = frame[station_text.str.contains(STATION_PATTERN, case=False, regex=True)].copy()
    if selected.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "station": selected[market_column].astype(str).str.strip(),
            "price_date": pd.Timestamp(year=year, month=month, day=1)
            + pd.to_timedelta((week - 1) * 7, unit="D"),
            "modal_price_per_quintal": pd.to_numeric(
                selected[price_column], errors="coerce"
            ),
            "district": selected[market_column].astype(str).str.strip(),
            "source_year": year,
            "source_month": month,
            "source_week": week,
        }
    ).dropna(subset=["modal_price_per_quintal"])


def main():
    parser = argparse.ArgumentParser(description="Fetch Maharashtra onion prices from AGMARKNET.")
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--state-id", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    failures = 0
    with requests.Session() as session:
        for year in range(args.start_year, args.end_year + 1):
            for month in range(1, 13):
                for week in range(1, 5):
                    try:
                        content = fetch_report(session, year, month, week, args.state_id)
                        raw_path = args.raw_dir / f"{year}-{month:02d}-week-{week}.xlsx"
                        raw_path.write_bytes(content)
                        normalized = normalize_report(content, year, month, week)
                        if not normalized.empty:
                            reports.append(normalized)
                    except (OSError, ValueError, requests.RequestException) as error:
                        failures += 1
                        print(f"FAILED {year}-{month:02d} week {week}: {error}")

    if not reports:
        raise RuntimeError(
            "No Nashik/Lasalgaon rows found. Inspect raw reports for a changed schema or station name."
        )
    result = pd.concat(reports, ignore_index=True)
    result = result.sort_values(["price_date", "station"]).drop_duplicates(
        ["price_date", "station"], keep="last"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} upstream rows to {args.output}")
    print(f"Stations: {sorted(result['station'].unique())}")
    print(f"Failed reports: {failures}")


if __name__ == "__main__":
    main()
