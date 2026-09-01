from agmarknet_api import AgmarknetClient
from pathlib import Path

client = AgmarknetClient()

OUTPUT_DIR = Path("data/raw/agmarknet/2026/onion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for month in range(1, 9):

    for week in range(1, 5):

        print(f"Downloading 2026-{month:02d} Week {week}")

        try:
            result = client.wholesale_prices_weekly(
                report_mode="Districtwise",
                commodity_id=23,
                year=2026,
                month=month,
                week=week,
                state_id=31,
                district_id=0,
                export=True,
            )

            output = (
                OUTPUT_DIR
                / f"2026-{month:02d}-week-{week}.xlsx"
            )

            output.write_bytes(result)

            print(f"Saved: {output}")

        except Exception as e:
            print(f"FAILED: {e}")