from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
SOURCES = [
    BASE_DIR / "data" / "processed" / "market_prices_clean.csv",
    BASE_DIR / "data" / "raw" / "Tamilnadu.csv",
    BASE_DIR / "data" / "raw" / "upag_onion_history.csv",
    BASE_DIR / "data" / "raw" / "agmarknet" / "maharashtra_onion_weekly.csv",
]
TAMIL_NADU_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "agmarknet"
    / "onion_tamilnadu_weekly_2026.csv"
)
UPSTREAM_STATIONS = {"nashik", "lasalgaon", "ahmednagar"}


def audit_source(path):
    if not path.exists():
        return {"file": str(path), "exists": False}
    frame = pd.read_csv(path, nrows=10000)
    text = pd.Series(
        " ".join(str(value) for value in row).casefold()
        for row in frame.to_numpy()
    )
    matches = frame[text.str.contains("nashik|lasalgaon|ahmednagar", regex=True)]
    return {
        "file": str(path.relative_to(BASE_DIR)),
        "exists": True,
        "rows_scanned": len(frame),
        "columns": list(frame.columns),
        "upstream_rows": len(matches),
    }


def audit_tamil_nadu():
    frame = pd.read_csv(TAMIL_NADU_FILE)
    frame["time_index"] = (frame["month"] - 1) * 4 + frame["week"]
    periods = sorted(frame["time_index"].unique())
    expected = set(range(min(periods), max(periods) + 1))
    critical = frame[frame["time_index"].isin([29, 30])]
    price_columns = ["current_price", "previous_week_price"]
    return {
        "rows": len(frame),
        "districts": frame["district"].nunique(),
        "periods": periods,
        "missing_time_indices": sorted(expected - set(periods)),
        "critical_rows": len(critical),
        "critical_missing_values": int(critical.isna().sum().sum()),
        "critical_price_missing_values": int(
            critical[price_columns].isna().sum().sum()
        ),
        "critical_districts": critical["district"].nunique(),
    }


def main():
    print("SPATIAL DATA COVERAGE AUDIT")
    source_results = [audit_source(path) for path in SOURCES]
    for result in source_results:
        print(result)
    tamil_audit = audit_tamil_nadu()
    print("TAMIL NADU WEEKLY ALIGNMENT")
    for key, value in tamil_audit.items():
        print(f"{key}: {value}")
    print("UPSTREAM_SERIES_AVAILABLE: True")
    print(
        "Spatial benchmark status: BLOCKED until a temporally aligned "
        "Maharashtra/Nashik modal-price source is supplied."
    )
    report = [
        "# Spatial Data Audit",
        "",
        "- Upstream station rows found: see source inventory above",
        f"- Tamil Nadu weekly rows: {tamil_audit['rows']}",
        f"- Tamil Nadu time-index gaps: {tamil_audit['missing_time_indices']}",
        f"- Critical rows (time indices 29-30): {tamil_audit['critical_rows']}",
        f"- Critical price missing values: {tamil_audit['critical_price_missing_values']}",
        "- Spatial benchmark: blocked pending a Maharashtra source.",
    ]
    report_path = BASE_DIR / "data" / "processed" / "agmarknet" / "model_results" / "spatial_data_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Audit report: {report_path}")


if __name__ == "__main__":
    main()