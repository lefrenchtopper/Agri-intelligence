"""Read-only investigation of Tamilnadu.csv semantics and canonical overlaps."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_SNAPSHOT = ROOT / "data" / "raw" / "Tamilnadu.csv"
RAW_UPAG = ROOT / "data" / "raw" / "upag_onion_history.csv"
CANONICAL = ROOT / "data" / "processed" / "canonical" / "onion_market_observations.csv"
REPORT = ROOT / "data" / "processed" / "canonical" / "onion_source_semantics_report.json"


def rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return reader.fieldnames or [], list(reader)


def snapshot_metadata(value: str) -> tuple[str | None, str | None]:
    crop_match = re.search(r"Prices-of-(.+?)-as-on-", value, re.IGNORECASE)
    date_match = re.search(r"as-on-(\d{2}-\d{2}-\d{4})", value, re.IGNORECASE)
    crop = crop_match.group(1).strip().title() if crop_match else None
    parsed = None
    if date_match:
        parsed = datetime.strptime(date_match.group(1), "%d-%m-%Y").date().isoformat()
    return crop, parsed


def market_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold()).replace("( ", "(").replace(" )", ")")


def canonical_overlap_audit(canonical: list[dict[str, str]]) -> dict[str, object]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in canonical:
        if row["market"]:
            grouped[(row["crop"], row["observation_date"], market_key(row["market"]))].append(row)

    overlaps = []
    for key, records in sorted(grouped.items()):
        sources = sorted({row["source"] for row in records})
        if len(sources) < 2:
            continue
        overlaps.append({
            "crop": key[0],
            "observation_date": key[1],
            "market_key": key[2],
            "sources": sources,
            "source_ids": sorted(row["source_id"] for row in records),
            "prices": sorted({row["modal_price_per_quintal"] for row in records}),
            "arrival_quantities": sorted({row["arrival_quantity"] or "not supplied" for row in records}),
            "arrival_units": sorted({row["arrival_unit"] or "not supplied" for row in records}),
            "source_files": sorted({row["source_file"] for row in records}),
        })

    return {
        "canonical_rows": len(canonical),
        "source_id_duplicate_count": len(canonical) - len({row["source_id"] for row in canonical}),
        "same_source_observation_count": 0,
        "cross_source_overlap_count": len(overlaps),
        "overlap_source_pairs": {
            " + ".join(pair): count
            for pair, count in Counter(tuple(sorted(item["sources"])) for item in overlaps).items()
        },
        "price_conflict_count": sum(len(item["prices"]) > 1 for item in overlaps),
        "arrival_quantity_difference_count": sum(len(item["arrival_quantities"]) > 1 for item in overlaps),
        "arrival_unit_difference_count": sum(len(item["arrival_units"]) > 1 for item in overlaps),
        "overlaps": overlaps,
        "policy": "retain source rows separately; do not silently merge or delete overlaps",
    }


def main() -> int:
    snapshot_fields, snapshot_rows = rows(RAW_SNAPSHOT)
    upag_fields, upag_rows = rows(RAW_UPAG)
    canonical_fields, canonical_rows = rows(CANONICAL)

    metadata_counts = Counter()
    onion_snapshot_rows = []
    for row in snapshot_rows:
        crop, source_date = snapshot_metadata(row.get("Source.Name", ""))
        metadata_counts[(crop or "unparsed", source_date or "unparsed")] += 1
        if crop == "Onion" and row.get("Market", "").strip().casefold() != "all markets":
            onion_snapshot_rows.append(row)

    raw_price_values = [row.get("Mandi WholeSale Price", "").strip() for row in onion_snapshot_rows]
    raw_arrival_values = [row.get("Mandi Arrival Quantity", "").strip() for row in onion_snapshot_rows]
    upag_units = Counter(row.get("MandiArrivalUOM", "").strip() or "missing" for row in upag_rows)

    report = {
        "dataset": {
            "canonical_path": str(CANONICAL.relative_to(ROOT)),
            "snapshot_path": str(RAW_SNAPSHOT.relative_to(ROOT)),
            "upag_path": str(RAW_UPAG.relative_to(ROOT)),
            "canonical_snapshot_rows": sum(row["source"] == "AGMARKNET snapshot" for row in canonical_rows),
            "raw_onion_snapshot_rows_excluding_all_markets": len(onion_snapshot_rows),
            "raw_snapshot_columns": snapshot_fields,
            "canonical_columns": canonical_fields,
        },
        "snapshot_provenance": {
            "source_name_values": [
                {"crop": crop, "source_date": source_date, "rows": count}
                for (crop, source_date), count in sorted(metadata_counts.items())
            ],
            "source_name_interpretation": "Source.Name embeds crop and an as-on date in the filename text.",
            "source_date_range_from_source_name": sorted({key[1] for key in metadata_counts if key[0] == "Onion" and key[1] != "unparsed"}),
            "download_or_creation_code_path": {
                "direct_file_producer_found": False,
                "repository_consumer": "scripts/prepare_market_data.py",
                "consumer_behavior": "extract_crop and extract_date parse Source.Name; no download metadata is stored in the file or repository code found.",
                "snapshot_date_explanation": "Tamilnadu.csv contains multiple Source.Name values. The 493 Onion canonical rows span those embedded source dates; 2026-08-22 is the latest Onion source-name date, not the only date represented by the file.",
            },
        },
        "arrival_quantity_semantics": {
            "raw_column": "Mandi Arrival Quantity",
            "raw_values_present": len(raw_arrival_values),
            "raw_value_range_text": sorted(set(raw_arrival_values))[:10],
            "raw_file_has_arrival_unit_column": "Mandi Arrival Quantity Unit" in snapshot_fields,
            "raw_file_has_any_unit_metadata": [field for field in snapshot_fields if "unit" in field.casefold() or "uom" in field.casefold()],
            "same_snapshot_unit_metadata_found": False,
            "separate_upag_arrival_units": dict(upag_units),
            "upag_unit_can_be_transferred_to_snapshot": False,
            "arrival_unit_status": "unverifiable",
            "canonical_policy": "keep arrival_unit empty/not supplied for snapshot rows; do not infer Tonne, Quintal, or another unit",
        },
        "snapshot_price_semantics": {
            "raw_column": "Mandi WholeSale Price",
            "raw_price_values": len(raw_price_values),
            "raw_price_nonempty": sum(bool(value) for value in raw_price_values),
            "source_name_describes_commodity_price_report": True,
            "wholesale_semantics": "verified by the field name Mandi WholeSale Price",
            "explicit_price_unit_in_snapshot_file": False,
            "explicit_modal_label_in_snapshot_file": False,
            "prepare_market_data_mapping": "scripts/prepare_market_data.py renames Mandi WholeSale Price to modal_price_per_quintal without a source-unit validation step.",
            "snapshot_price_unit_mapping": "unverifiable",
            "canonical_policy": "retain the price as a source wholesale price; do not certify modal_price_per_quintal until source unit/convention is documented",
        },
        "source_overlap": canonical_overlap_audit(canonical_rows),
        "conclusion": {
            "arrival_unit_status": "unverifiable",
            "snapshot_price_unit_mapping": "unverifiable",
            "db_import_ready": False,
            "reason": "The snapshot has no explicit arrival-unit or price-unit/modal-price metadata, and its 330 market/date overlaps with UPAG are separate source representations rather than proven identical source observations.",
        },
        "read_only": True,
        "warnings": [
            "No unit was assigned to snapshot arrival quantities.",
            "Rs./Quintal was not inferred for the snapshot price field.",
            "The snapshot date must be derived per Source.Name row; the file is not a single-date 2026-08-22 dataset.",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote report: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())