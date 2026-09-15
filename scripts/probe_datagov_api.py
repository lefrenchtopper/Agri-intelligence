from pathlib import Path
import pandas as pd
import requests
import urllib3

# Suppress SSL warnings for testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
DEMO_KEY = "579b464db66ec23bdd000001cdd39463285847b34cb4574fe9263864"

output_dir = Path("data/raw/datagov_probe")
output_dir.mkdir(parents=True, exist_ok=True)

test_years = ["2021", "2022", "2023"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

print(f"{'YEAR':<6} | {'STATUS':<6} | {'TOTAL RECORDS':<13} | {'SAMPLE MARKET'}")
print("-" * 55)

for year in test_years:
    params = {
        "api-key": DEMO_KEY,
        "format": "json",
        "limit": 10,
        "filters[state.keyword]": "Tamil Nadu",
        "filters[commodity.keyword]": "Onion",
        "filters[arrival_date]": f"15/03/{year}",
    }

    try:
        r = requests.get(
            API_URL,
            params=params,
            headers=HEADERS,
            verify=False,  # Bypass SSL handshake block
            timeout=25,
        )

        if r.status_code == 200:
            data = r.json()
            total = data.get("total", 0)
            records = data.get("records", [])

            sample_market = (
                records[0].get("market", "N/A") if records else "NO DATA"
            )
            print(
                f"{year:<6} | {r.status_code:<6} | {total:<13} | {sample_market}"
            )

            if records:
                df = pd.DataFrame(records)
                df.to_csv(
                    output_dir / f"datagov_tn_onion_{year}.csv", index=False
                )
        else:
            print(f"{year:<6} | {r.status_code:<6} | {'0':<13} | HTTP ERROR")

    except Exception as e:
        print(f"{year:<6} | ERROR  | {'0':<13} | {str(e)[:30]}")