from io import BytesIO
from pathlib import Path

import pandas as pd
from agmarknet_api import AgmarknetClient

client = AgmarknetClient()

ENDPOINT = "/price-trend/wholesale-prices-weekly"
OUTPUT_DIR = Path("data/raw/agmarknet/marketwise_probe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tests = [
    (2021, 3, 1),
    (2021, 7, 1),
    (2021, 11, 1),
    (2022, 3, 1),
    (2022, 7, 1),
    (2022, 11, 1),
    (2023, 3, 1),
    (2023, 7, 1),
    (2023, 11, 1),
    (2024, 3, 1),
    (2024, 7, 1),
    (2024, 11, 1),
]

print(
    f"{'DATE':<12} | {'STATUS':<6} | {'BYTES':<7} | "
    f"{'ROWS':<6} | {'COIMBATORE':<12}"
)
print("-" * 65)

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

    rows = 0
    coimbatore = 0

    if response.status_code == 200:
        try:
            sheets = pd.read_excel(
                BytesIO(response.content),
                sheet_name=None,
                header=None,
            )

            for df in sheets.values():
                if df.empty:
                    continue

                rows += len(df)

                matches = df.astype(str).apply(
                    lambda col: col.str.contains(
                        "coimbatore",
                        case=False,
                        na=False,
                    )
                )

                coimbatore += int(matches.any(axis=1).sum())

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
                f"PARSE ERROR | {exc}"
            )
            continue

    print(
        f"{year}-{month:02d}-W{week:<2} | "
        f"{response.status_code:<6} | "
        f"{len(response.content):<7} | "
        f"{rows:<6} | "
        f"{coimbatore:<12}"
    )