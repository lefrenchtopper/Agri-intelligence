import time
import requests

# Switched from http:// to https://
BASE_URL = "https://api.agmarknet.gov.in/v1/price-trend/wholesale-prices-weekly"

COMMODITY_ONION = 23
STATE_TN = 31
DISTRICT_COIMBATORE = 530

YEARS_TO_TEST = [2021, 2022, 2023]
SAMPLE_MONTHS = [1, 6, 12]
SAMPLE_WEEKS = [1, 3]

# Domain headers required by Agmarknet firewall
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://agmarknet.gov.in/",
    "Origin": "https://agmarknet.gov.in"
}

print("--- Starting 2021-2023 Agmarknet Historical Coverage Discovery ---")

total_checks = 0
successful_checks = 0

for year in YEARS_TO_TEST:
    for month in SAMPLE_MONTHS:
        for week in SAMPLE_WEEKS:
            total_checks += 1
            params = {
                "report_mode": "Marketwise",
                "commodity": COMMODITY_ONION,
                "year": year,
                "month": month,
                "week": week,
                "state": STATE_TN,
                "district": DISTRICT_COIMBATORE,
                "export": "true"
            }
            
            try:
                response = requests.get(BASE_URL, params=params, headers=headers, timeout=10)
                content_length = len(response.content)
                
                if response.status_code == 200 and content_length > 500:
                    successful_checks += 1
                    print(f"✓ [{year}-M{month:02d}-W{week}] HTTP 200 | Size: {content_length} bytes | DATA AVAILABLE")
                else:
                    print(f"✗ [{year}-M{month:02d}-W{week}] HTTP {response.status_code} | Size: {content_length} bytes | FAILED")
            
            except Exception as e:
                print(f"✗ [{year}-M{month:02d}-W{week}] ERROR: {str(e)}")
            
            time.sleep(1)

print(f"\nSummary: {successful_checks} / {total_checks} successful connections.")