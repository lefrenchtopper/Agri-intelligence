from pathlib import Path
import time
import pandas as pd
import requests

API_URL = "https://api.agmarknet.gov.in/v1/price-trend/wholesale-prices-weekly"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101"
        " Firefox/128.0"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://agmarknet.gov.in/",
    "Origin": "https://agmarknet.gov.in",
}


def parse_agmarknet_payload(
    res_json: dict, year: int, month: int, week: int
) -> list:
  """Extracts row objects and maps dynamic price keys to standard column headers."""
  if not res_json.get("success") or "rows" not in res_json:
    return []

  rows = res_json.get("rows", [])
  columns_meta = res_json.get("columns", [])

  # 1. Dynamically extract price keys from column metadata
  price_keys = []
  for col_group in columns_meta:
    if col_group.get("key") == "prices":
      price_keys = [c["key"] for c in col_group.get("columns", [])]
      break

  parsed_records = []

  # 2. Iterate through rows (35 districts + 1 Average row)
  for r in rows:
    district_name = r.get("district", "").strip()

    # Distinguish district rows from aggregate summary
    is_average = district_name.lower() == "average"

    record = {
        "query_year": year,
        "query_month": month,
        "query_week": week,
        "district": district_name,
        "record_type": (
            "State Average Summary" if is_average else "District Level"
        ),
    }

    # Map dynamic price keys to fixed column names
    if len(price_keys) >= 4:
      record["price_current_week"] = r.get(price_keys[0])
      record["price_prev_week"] = r.get(price_keys[1])
      record["price_prev_month"] = r.get(price_keys[2])
      record["price_prev_year"] = r.get(price_keys[3])

    # Percentage change values
    record["change_prev_week_pct"] = r.get("change_over_previous_week")
    record["change_prev_month_pct"] = r.get("change_over_previous_month")
    record["change_prev_year_pct"] = r.get("change_over_previous_year")

    parsed_records.append(record)

  return parsed_records


def fetch_full_year_agmarknet(
    year: int = 2026, commodity_id: int = 23, state_id: int = 31
) -> pd.DataFrame:
  all_records = []

  print(
      f"Fetching full year data for Year: {year} | Commodity: {commodity_id} |"
      f" State: {state_id}..."
  )

  for month in range(1, 13):
    for week in range(1, 5):
      params = {
          "report_mode": "Districtwise",
          "commodity": commodity_id,
          "year": year,
          "month": month,
          "week": week,
          "state": state_id,
          "district": 0,
          "export": "false",
      }

      try:
        response = requests.get(
            API_URL, headers=HEADERS, params=params, timeout=15
        )
        if response.status_code == 200:
          res_json = response.json()
          records = parse_agmarknet_payload(res_json, year, month, week)

          if records:
            all_records.extend(records)
            print(
                f"  ✓ Month {month:02d}, Week {week}: Extracted {len(records)}"
                " rows"
            )
          else:
            print(f"  - Month {month:02d}, Week {week}: No data returned")
        else:
          print(
              f"  ✗ Month {month:02d}, Week {week}: Failed (HTTP"
              f" {response.status_code})"
          )
      except Exception as err:
        print(f"  ! Month {month:02d}, Week {week}: Error ({err})")

      time.sleep(0.3)  # Polite request rate limiting

  return pd.DataFrame(all_records)


if __name__ == "__main__":
  df_result = fetch_full_year_agmarknet(
      year=2026, commodity_id=23, state_id=31
  )

  if not df_result.empty:
    output_dir = Path("data/agmarknet_weekly")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full consolidated file
    full_output = output_dir / "tn_onion_weekly_2026_full.csv"
    df_result.to_csv(full_output, index=False)

    # Separate District-level data and State Average summaries
    df_districts = df_result[df_result["record_type"] == "District Level"]
    df_averages = df_result[
        df_result["record_type"] == "State Average Summary"
    ]

    df_districts.to_csv(
        output_dir / "tn_onion_weekly_2026_districts.csv", index=False
    )
    df_averages.to_csv(
        output_dir / "tn_onion_weekly_2026_state_averages.csv", index=False
    )

    print("\n--- FETCHING & PARSING COMPLETE ---")
    print(f"Total Records Extracted : {len(df_result)}")
    print(f" ├── District Records   : {len(df_districts)}")
    print(f" └── Summary Averages   : {len(df_averages)}")
    print(f"Saved to: {full_output}")
  else:
    print("No records retrieved.")