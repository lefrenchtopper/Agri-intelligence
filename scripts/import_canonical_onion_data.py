"""Normalize real Tamil Nadu Onion observations into a canonical CSV.

This is a staging importer. It intentionally does not write to SQLite because
the current schema cannot preserve source IDs, units, report periods, or source
files and the existing database contains synthetic market rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "canonical" / "onion_market_observations.csv"
MONTHS = {
    name.lower(): number
    for number, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"),
        start=1,
    )
}
MONTHS.update({name[:3].lower(): number for name, number in MONTHS.copy().items()})


@dataclass(frozen=True)
class Observation:
    source_id: str
    crop: str
    state: str
    observation_date: date
    period_start: date
    period_end: date
    district: str
    market: str
    modal_price_per_quintal: float
    arrival_quantity: str
    arrival_unit: str
    price_unit: str
    source: str
    source_file: str
    source_row: int
    source_price_column: str

    def as_row(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "crop": self.crop,
            "state": self.state,
            "observation_date": self.observation_date.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "district": self.district,
            "market": self.market,
            "modal_price_per_quintal": f"{self.modal_price_per_quintal:.2f}",
            "arrival_quantity": self.arrival_quantity,
            "arrival_unit": self.arrival_unit,
            "price_unit": self.price_unit,
            "source": self.source,
            "source_file": self.source_file,
            "source_row": self.source_row,
            "source_price_column": self.source_price_column,
        }


def parse_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()[:10]
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_price(value: object) -> float | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9.-]", "", str(value).strip())
    if not cleaned:
        return None
    try:
        price = float(cleaned)
    except ValueError:
        return None
    return price if price > 0 else None


def parse_quantity(value: object) -> str:
    if value is None or not str(value).strip():
        return ""
    return str(value).strip()


def source_id(*parts: object) -> str:
    payload = "|".join(str(part).strip() for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def add_observation(observations: list[Observation], counters: Counter[str], **kwargs: object) -> None:
    price = parse_price(kwargs.pop("price"))
    if price is None:
        counters["invalid_prices"] += 1
        return
    start = kwargs.pop("period_start")
    end = kwargs.pop("period_end")
    if not isinstance(start, date) or not isinstance(end, date):
        counters["invalid_dates"] += 1
        return
    if end < start:
        counters["invalid_dates"] += 1
        return
    unit = normalize_name(kwargs.pop("price_unit"))
    if not unit or "qtl" not in unit.casefold() and "quintal" not in unit.casefold():
        counters["invalid_units"] += 1
        return
    observation = Observation(
        source_id=source_id(kwargs["source_file"], kwargs["source_row"], kwargs["source_price_column"], kwargs["district"], kwargs["market"], start, end, price),
        modal_price_per_quintal=price,
        period_start=start,
        period_end=end,
        observation_date=end,
        **kwargs,
        price_unit=unit,
    )
    observations.append(observation)
    counters["valid"] += 1


def load_upag(path: Path, observations: list[Observation], counters: Counter[str]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row_number, row in enumerate(csv.DictReader(source), start=2):
            record_date = parse_date(row.get("record_date") or row.get("CalendarDay"))
            if record_date is None:
                counters["invalid_dates"] += 1
                continue
            if normalize_name(row.get("Commodity")).casefold() != "onion":
                counters["non_onion"] += 1
                continue
            add_observation(
                observations,
                counters,
                crop="Onion",
                state=normalize_name(row.get("StateName")),
                period_start=record_date,
                period_end=record_date,
                district=normalize_name(row.get("District")),
                market=normalize_name(row.get("Market")),
                price=row.get("MandiWholeSalePrice"),
                arrival_quantity=parse_quantity(row.get("MandiArrivalQuantity")),
                arrival_unit=normalize_name(row.get("MandiArrivalUOM")),
                price_unit=normalize_name(row.get("UOM")),
                source="UPAG",
                source_file=str(path.relative_to(ROOT)),
                source_row=row_number,
                source_price_column="MandiWholeSalePrice",
            )


def snapshot_date(source_name: str) -> date | None:
    match = re.search(r"as-on-(\d{2}-\d{2}-\d{4})", source_name, re.IGNORECASE)
    return parse_date(match.group(1)) if match else None


def load_tamilnadu_snapshot(path: Path, observations: list[Observation], counters: Counter[str]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row_number, row in enumerate(csv.DictReader(source), start=2):
            source_name = normalize_name(row.get("Source.Name"))
            crop_match = re.search(r"Prices-of-(.+?)-as-on-", source_name, re.IGNORECASE)
            if not crop_match or crop_match.group(1).strip().casefold() != "onion":
                counters["non_onion"] += 1
                continue
            record_date = snapshot_date(row.get("Source.Name", ""))
            if record_date is None:
                counters["invalid_dates"] += 1
                continue
            market = normalize_name(row.get("Market"))
            if market.casefold() == "all markets":
                counters["aggregate_rows_excluded"] += 1
                continue
            add_observation(
                observations,
                counters,
                crop="Onion",
                state=normalize_name(row.get("StateName")),
                period_start=record_date,
                period_end=record_date,
                district="",
                market=market,
                price=row.get("Mandi WholeSale Price"),
                arrival_quantity=parse_quantity(row.get("Mandi Arrival Quantity")),
                arrival_unit="",
                price_unit="Rs./Quintal",
                source="AGMARKNET snapshot",
                source_file=str(path.relative_to(ROOT)),
                source_row=row_number,
                source_price_column="Mandi WholeSale Price",
            )


def parse_report_period(column: str) -> tuple[date, date] | None:
    match = re.search(r"Prices\s+(\d{1,2})-(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", column)
    if not match:
        return None
    start_day, end_day, month_name, year = match.groups()
    month = MONTHS.get(month_name.casefold())
    if month is None:
        return None
    return date(int(year), month, int(start_day)), date(int(year), month, int(end_day))


def load_agmarknet_reports(root: Path, observations: list[Observation], counters: Counter[str]) -> None:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("pandas is required to read AGMARKNET Excel reports") from error

    files = sorted((root / "data" / "raw" / "agmarknet" / "2026").rglob("*.xlsx"))
    counters["agmarknet_files"] = len(files)
    for path in files:
        frame = pd.read_excel(path, header=1)
        price_columns = [str(column).strip() for column in frame.columns if str(column).strip().startswith("Prices")]
        if len(price_columns) < 1:
            counters["report_errors"] += 1
            continue
        current_column = price_columns[0]
        period = parse_report_period(current_column)
        if period is None:
            counters["report_errors"] += 1
            continue
        start, end = period
        for row_number, row in frame.iterrows():
            district = normalize_name(row.get("District"))
            if not district or district.casefold() in {"average", "note:"}:
                counters["aggregate_rows_excluded"] += 1
                continue
            add_observation(
                observations,
                counters,
                crop="Onion",
                state="Tamil Nadu",
                period_start=start,
                period_end=end,
                district=district,
                market="",
                price=row.get(current_column),
                arrival_quantity="",
                arrival_unit="",
                price_unit="Rs./Quintal",
                source="AGMARKNET weekly district report",
                source_file=str(path.relative_to(ROOT)),
                source_row=int(row_number) + 3,
                source_price_column=current_column,
            )


def write_csv(path: Path, observations: list[Observation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(observations[0].as_row()) if observations else list(Observation.__annotations__)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(observation.as_row() for observation in observations)


def report(observations: list[Observation], counters: Counter[str], output: Path) -> None:
    dates = [observation.observation_date for observation in observations]
    keys = [
        (observation.crop, observation.observation_date, observation.district, observation.market, observation.modal_price_per_quintal)
        for observation in observations
    ]
    duplicate_count = len(keys) - len(set(keys))
    periods = sorted({(observation.observation_date.year, observation.observation_date.month, observation.observation_date.isocalendar().week) for observation in observations})
    years = sorted({period[0] for period in periods})
    districts = {observation.district for observation in observations if observation.district}
    markets = {observation.market for observation in observations if observation.market}
    missing_weeks = 0
    if dates:
        first_week = min(dates).isocalendar().week
        last_week = max(dates).isocalendar().week
        missing_weeks = max(0, (max(dates) - min(dates)).days // 7 + 1 - len({observation.observation_date.isocalendar()[:2] for observation in observations}))

    print("CANONICAL REAL ONION IMPORT REPORT")
    print("mode: real-source normalization only; SQLite and model artifacts untouched")
    print(f"output: {output.relative_to(ROOT)}")
    print("crop: Onion")
    print("state: Tamil Nadu")
    print(f"earliest observation: {min(dates) if dates else 'none'}")
    print(f"latest observation: {max(dates) if dates else 'none'}")
    print(f"years covered: {years}")
    print(f"periods covered (year/month/ISO week): {len(periods)}")
    print(f"observations: {len(observations)}")
    print(f"districts: {len(districts)}")
    print(f"markets: {len(markets)}")
    print(f"missing ISO-week slots between first and last observed dates: {missing_weeks}")
    print(f"duplicate observations: {duplicate_count}")
    print(f"duplicate observations excluded from output: {counters['duplicates_excluded']}")
    print(f"invalid dates: {counters['invalid_dates']}")
    print(f"invalid prices: {counters['invalid_prices']}")
    print(f"invalid units: {counters['invalid_units']}")
    print(f"aggregate rows excluded: {counters['aggregate_rows_excluded']}")
    print(f"AGMARKNET Excel reports read: {counters['agmarknet_files']}")
    print(f"AGMARKNET report errors: {counters['report_errors']}")
    print(f"source counts: {Counter(observation.source for observation in observations)}")
    print(f"coverage by year: {Counter(observation.observation_date.year for observation in observations)}")
    print(f"coverage by year/month: {Counter((observation.observation_date.year, observation.observation_date.month) for observation in observations)}")
    print(f"coverage by year/month/ISO week: {Counter((observation.observation_date.year, observation.observation_date.month, observation.observation_date.isocalendar().week) for observation in observations)}")
    print("database inserts: 0")
    print("synthetic data used: no")


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize real Tamil Nadu Onion sources without database insertion.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output

    observations: list[Observation] = []
    counters: Counter[str] = Counter()
    load_upag(ROOT / "data" / "raw" / "upag_onion_history.csv", observations, counters)
    load_tamilnadu_snapshot(ROOT / "data" / "raw" / "Tamilnadu.csv", observations, counters)
    load_agmarknet_reports(ROOT, observations, counters)

    observations.sort(key=lambda item: (item.observation_date, item.district, item.market, item.source_file, item.source_row))
    unique_observations: list[Observation] = []
    seen_keys: set[tuple[object, ...]] = set()
    for observation in observations:
        key = (
            observation.crop,
            observation.observation_date,
            observation.district,
            observation.market,
            observation.modal_price_per_quintal,
        )
        if key in seen_keys:
            counters["duplicates_excluded"] += 1
            continue
        seen_keys.add(key)
        unique_observations.append(observation)
    observations = unique_observations
    write_csv(output, observations)
    report(observations, counters, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())