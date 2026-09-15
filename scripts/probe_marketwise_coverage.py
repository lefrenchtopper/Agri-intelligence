from io import BytesIO
from pathlib import Path

import pandas as pd
from agmarknet_api import AgmarknetClient

client = AgmarknetClient()

ENDPOINT = "/price-trend/wholesale-prices-weekly"
OUTPUT_DIR = Path("data/raw/agmarknet/marketwise_coverage")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tests = []

for year in [2023, 2024]:
    for month in range(1, 13):
        for week in [1, 2, 3, 4]:
            tests.append((year, month, week))

print(
    f"{'DATE':<14} | {'STATUS':<6} | {'BYTES':<7} | "
    f"{'DATA_ROWS':<10} | {'COIMBATORE_MATCH':<16}"
)
print("-" * 75)

for year, month, week in tests:
    params = {
        "report_mode": "MarketwiseAll",
        "commodity": 23,
        "year": year,
        "month": month,
        "week": week,
        "state": 31,
        "district": 0,
        "market": 0,
        "export": "true",
    }

    response = client.session.get(
        client._url(ENDPOINT),
        params=params,
        timeout=client.timeout,
    )

    if response.status_code != 200:
        print(
            f"{year}-{month:02d}-W{week:<2} | "
            f"{response.status_code:<6} | "
            f"{len(response.content):<7} | "
            f"HTTP ERROR"
        )
        continue

    try:
        sheets = pd.read_excel(
            BytesIO(response.content),
            sheet_name=None,
            header=None,
        )

        data_rows = 0
        coimbatore_matches = []

        for df in sheets.values():
            if df.empty:
                continue

            for _, row in df.iterrows():
                first = str(row.iloc[0]).strip()

                if (
                    first
                    and first.lower() not in {
                        "nan",
                        "average",
                    }
                    and not first.lower().startswith("note:")
                    and not first.lower().startswith(
                        "weighted average price"
                    )
                    and not first.lower().startswith(
                        "change (over"
                    )
                    and "market prices" not in first.lower()
                    and "market-wise wholesale" not in first.lower()
                ):
                    data_rows += 1

                    if "coimbatore" in first.lower():
                        coimbatore_matches.append(first)

        print(
            f"{year}-{month:02d}-W{week:<2} | "
            f"{response.status_code:<6} | "
            f"{len(response.content):<7} | "
            f"{data_rows:<10} | "
            f"{len(coimbatore_matches):<16}"
        )

        if data_rows > 5:
            output_file = (
                OUTPUT_DIR
                / f"marketwise_{year}_{month:02d}_w{week}.xlsx"
            )
            output_file.write_bytes(response.content)

    except Exception as exc:
        print(
            f"{year}-{month:02d}-W{week:<2} | "
            f"{response.status_code:<6} | "
            f"{len(response.content):<7} | "
            f"PARSE ERROR: {exc}"
        )