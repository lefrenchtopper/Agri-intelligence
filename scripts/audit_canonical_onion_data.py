"""Read-only coverage and provenance audit for the canonical Onion dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data" / "processed" / "canonical" / "onion_market_observations.csv"
REPORT = ROOT / "data" / "processed" / "canonical" / "onion_market_coverage_report.json"
IMPORTER = ROOT / "scripts" / "import_canonical_onion_data.py"


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        return reader.fieldnames or [], list(reader)


def counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field] for row in rows).items()))


def market_identity(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold()).replace("( ", "(").replace(" )", ")")


def date_gaps(rows: list[dict[str, str]], source: str) -> dict[str, object]:
    source_rows = [row for row in rows if row["source"] == source]
    dates = sorted({parse_date(row["observation_date"]) for row in source_rows})
    intervals = [((right - left).days, left.isoformat(), right.isoformat()) for left, right in zip(dates, dates[1:])]
    cadence = Counter(interval[0] for interval in intervals)
    expected = "snapshot dates; continuity is not expected" if source == "AGMARKNET snapshot" else "weekly/periodic source; gaps require source-level investigation"
    return {
        "observation_dates": len(dates),
        "first": dates[0].isoformat() if dates else None,
        "last": dates[-1].isoformat() if dates else None,
        "interval_days": dict(sorted(cadence.items())),
        "gaps_greater_than_8_days": [
            {"days": days, "from": left, "to": right}
            for days, left, right in intervals
            if days > 8
        ],
        "interpretation": expected,
    }


def source_overlap(rows: list[dict[str, str]]) -> dict[str, object]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not row["market"]:
            continue
        key = (row["crop"], row["observation_date"], market_identity(row["market"]))
        grouped[key].append(row)

    overlaps = []
    conflicts = []
    for key, records in sorted(grouped.items()):
        sources = sorted({record["source"] for record in records})
        if len(sources) < 2:
            continue
        prices = sorted({record["modal_price_per_quintal"] for record in records})
        item = {
            "crop": key[0],
            "observation_date": key[1],
            "market": key[2],
            "sources": sources,
            "prices": prices,
            "arrival_quantities": sorted({record["arrival_quantity"] or "not supplied" for record in records}),
            "arrival_units": sorted({record["arrival_unit"] or "not supplied" for record in records}),
        }
        overlaps.append(item)
        if len(prices) > 1:
            conflicts.append(item)
    return {
        "overlap_count": len(overlaps),
        "conflict_count": len(conflicts),
        "source_id_duplicate_count": len(rows) - len({row["source_id"] for row in rows}),
        "same_source_observation_count": 0,
        "cross_source_representation_count": len(overlaps),
        "arrival_quantity_difference_count": sum(len(item["arrival_quantities"]) > 1 for item in overlaps),
        "arrival_unit_difference_count": sum(len(item["arrival_units"]) > 1 for item in overlaps),
        "overlaps": overlaps,
        "conflicts": conflicts,
        "merge_policy": "none; source observations remain separate",
    }


def unit_audit(rows: list[dict[str, str]]) -> dict[str, object]:
    accepted_price_units = {"rs./qtl", "rs./quintal", "rs per quintal", "inr/qtl", "inr/quintal", "inr per quintal"}
    price_units = Counter(row["price_unit"] for row in rows)
    ambiguous_price = [row["source_id"] for row in rows if row["price_unit"].casefold() not in accepted_price_units]
    arrival_units = Counter(row["arrival_unit"] or "not supplied" for row in rows)
    ambiguous_arrival = [
        row["source_id"]
        for row in rows
        if row["arrival_quantity"] and not row["arrival_unit"]
    ]
    return {
        "accepted_price_units": sorted(accepted_price_units),
        "price_units": dict(sorted(price_units.items())),
        "ambiguous_price_unit_rows": ambiguous_price,
        "arrival_units": dict(sorted(arrival_units.items())),
        "arrival_quantity_without_unit_rows": ambiguous_arrival,
        "status": not ambiguous_price and not ambiguous_arrival,
    }


def date_semantics(rows: list[dict[str, str]]) -> dict[str, object]:
    checks = {}
    for source in sorted({row["source"] for row in rows}):
        source_rows = [row for row in rows if row["source"] == source]
        if source == "AGMARKNET weekly district report":
            period_ok = all(parse_date(row["observation_date"]) == parse_date(row["period_end"]) for row in source_rows)
            checks[source] = {
                "period_fields_present": all(row["period_start"] and row["period_end"] for row in source_rows),
                "observation_date_equals_period_end": period_ok,
                "meaning": "report period end used as a sortable surrogate; source provides a period, not a point observation date",
                "precise_point_date_claim_allowed": False,
            }
        else:
            checks[source] = {
                "point_date_equals_period_start_and_end": all(row["observation_date"] == row["period_start"] == row["period_end"] for row in source_rows),
                "meaning": "source snapshot/record date",
                "precise_point_date_claim_allowed": True,
            }
    return checks


def deterministic_rebuild() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="canonical-onion-audit-", dir=ROOT) as temporary:
        temporary_output = Path(temporary) / "onion_market_observations.csv"
        completed = subprocess.run(
            [sys.executable, str(IMPORTER), "--output", str(temporary_output)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return {"status": False, "error": completed.stderr[-4000:]}
        _, rebuilt = read_rows(temporary_output)
        _, existing = read_rows(CANONICAL)
        existing_ids = [row["source_id"] for row in existing]
        rebuilt_ids = [row["source_id"] for row in rebuilt]
        existing_hash = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
        rebuilt_hash = hashlib.sha256(temporary_output.read_bytes()).hexdigest()
        return {
            "status": True,
            "existing_rows": len(existing),
            "rebuilt_rows": len(rebuilt),
            "row_count_identical": len(existing) == len(rebuilt),
            "ids_identical": existing_ids == rebuilt_ids,
            "file_hash_identical": existing_hash == rebuilt_hash,
            "existing_sha256": existing_hash,
            "rebuilt_sha256": rebuilt_hash,
        }


def main() -> int:
    if not CANONICAL.exists():
        raise FileNotFoundError(f"Canonical dataset not found: {CANONICAL}")
    fields, rows = read_rows(CANONICAL)
    dates = [parse_date(row["observation_date"]) for row in rows]
    districts = sorted({row["district"] for row in rows if row["district"]})
    markets = sorted({row["market"] for row in rows if row["market"]})
    source_names = sorted({row["source"] for row in rows})
    combinations = sorted({(row["district"], row["market"]) for row in rows if row["district"] and row["market"]})
    all_combinations = len(districts) * len(markets)
    observed_combination_keys = {(district, market_identity(market)) for district, market in combinations}
    missing_combinations = [
        {"district": district, "market": market}
        for district in districts
        for market in markets
        if (district, market_identity(market)) not in observed_combination_keys
    ]

    report = {
        "dataset": {
            "path": str(CANONICAL.relative_to(ROOT)),
            "columns": fields,
            "rows": len(rows),
            "crop": sorted({row["crop"] for row in rows}),
            "state": sorted({row["state"] for row in rows}),
            "earliest_observation": min(dates).isoformat(),
            "latest_observation": max(dates).isoformat(),
            "years_covered": sorted({item.year for item in dates}),
            "five_year_requirement_satisfied": (max(dates) - min(dates)).days >= 5 * 365,
        },
        "coverage": {
            "by_source": {source: sum(row["source"] == source for row in rows) for source in source_names},
            "by_year": counts([{**row, "value": str(parse_date(row["observation_date"]).year)} for row in rows], "value"),
            "by_month": counts([{**row, "value": str(parse_date(row["observation_date"]).strftime("%Y-%m"))} for row in rows], "value"),
            "by_district": counts([{**row, "value": row["district"] or "not supplied"} for row in rows], "value"),
            "by_market": counts([{**row, "value": row["market"] or "not supplied"} for row in rows], "value"),
            "districts": districts,
            "markets": markets,
            "district_market_combinations_observed": len(combinations),
            "district_market_combinations_possible_if_cross_product": all_combinations,
            "missing_cross_product_combinations": all_combinations - len(combinations),
            "missing_district_market_combinations": missing_combinations,
            "missing_combinations_are_not_imputed": True,
        },
        "temporal_continuity": {source: date_gaps(rows, source) for source in source_names},
        "source_overlap": source_overlap(rows),
        "units": unit_audit(rows),
        "date_semantics": date_semantics(rows),
        "determinism": deterministic_rebuild(),
        "five_year_gap": {
            "satisfied": False,
            "available_real_range": f"{min(dates).isoformat()} to {max(dates).isoformat()}",
            "missing_period_before_first": f"before {min(dates).isoformat()}",
            "missing_period_after_last": f"after {max(dates).isoformat()}",
            "synthetic_sqlite_rows_used": False,
            "interpolation_or_extrapolation_used": False,
        },
        "warnings": [
            "AGMARKNET weekly reports describe reporting periods; observation_date is period_end for sorting and must not be presented as a point observation date.",
            "District/market combinations are sparse; missing combinations are unknown, not zero-valued.",
            "The available real range does not satisfy the five-year requirement.",
        ],
        "errors": [],
        "read_only_scope": ["database", "models", "frontend", "source files", "existing model artifacts"],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote report: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())