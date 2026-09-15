import os
import time
import requests

BASE_URL = "https://api.agmarknet.gov.in/v1/price-trend/wholesale-prices-weekly"
OUTPUT_BASE_DIR = os.path.join("data", "raw", "agmarknet")

COMMODITY_ONION = 23
STATE_TN = 31
DISTRICT_COIMBATORE = 530

YEARS = [2026]
MONTH_MAP = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr",
    5: "may", 6: "jun", 7: "jul", 8: "aug",
    9: "sep", 10: "oct", 11: "nov", 12: "dec"
}
WEEKS = [1, 2, 3, 4]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://agmarknet.gov.in/",
    "Origin": "https://agmarknet.gov.in"
}

print("=== Starting Bulk Raw Download (2021 - 2025) ===")

total_downloaded = 0
total_skipped = 0
total_failed = 0

for year in YEARS:
    for month_num, month_name in MONTH_MAP.items():
        # Ensure target folder exists (e.g., data/raw/agmarknet/2021/jan)
        folder_path = os.path.join(OUTPUT_BASE_DIR, str(year), month_name)
        os.makedirs(folder_path, exist_ok=True)
        
        for week in WEEKS:
            file_name = f"week_{week}.csv"
            file_path = os.path.join(folder_path, file_name)
            
            # Skip if file already downloaded
            if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
                print(f"[SKIP] {year}/{month_name}/week_{week}.csv already exists.")
                total_skipped += 1
                continue
                
            params = {
                "report_mode": "Marketwise",
                "commodity": COMMODITY_ONION,
                "year": year,
                "month": month_num,
                "week": week,
                "state": STATE_TN,
                "district": DISTRICT_COIMBATORE,
                "export": "true"
            }
            
            try:
                response = requests.get(BASE_URL, params=params, headers=headers, timeout=15)
                
                if response.status_code == 200 and len(response.content) > 500:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    print(f"✓ [SAVED] {year}/{month_name}/week_{week}.csv ({len(response.content)} bytes)")
                    total_downloaded += 1
                else:
                    print(f"✗ [FAILED] {year}/{month_name}/week_{week}.csv | Status: {response.status_code}")
                    total_failed += 1
                    
            except Exception as e:
                print(f"✗ [ERROR] {year}/{month_name}/week_{week}.csv | {str(e)}")
                total_failed += 1
                
            time.sleep(1)  # Respect server rate limits

print("\n=== Download Complete Summary ===")
print(f"Files Downloaded: {total_downloaded}")
print(f"Files Skipped:    {total_skipped}")
print(f"Failed Requests:  {total_failed}")