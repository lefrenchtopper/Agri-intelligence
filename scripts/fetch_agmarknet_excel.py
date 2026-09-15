import os
import requests

URL = "https://api.agmarknet.gov.in/v1/price-trend/wholesale-prices-weekly"

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://agmarknet.gov.in",
    "Referer": "https://agmarknet.gov.in/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:155.0) Gecko/20100101 Firefox/155.0"
}

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Tamilnadu.xlsx")

def download_export(year: int = 2023, month: int = 1, week: int = 1, commodity: int = 23, state: int = 31):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    params = {
        "report_mode": "MarketwiseAll",
        "commodity": commodity,
        "year": year,
        "month": month,
        "week": week,
        "state": state,
        "export": "true"  # Triggers file download mode
    }
    
    response = requests.get(URL, headers=HEADERS, params=params, timeout=30)
    
    if response.status_code == 200 and len(response.content) > 0:
        with open(OUTPUT_FILE, "wb") as f:
            f.write(response.content)
        print(f"✅ Successfully saved export to '{OUTPUT_FILE}' ({len(response.content)} bytes).")
        return True
    else:
        print(f"❌ Failed to download file. Status: {response.status_code}")
        return False

if __name__ == "__main__":
    # Test downloading weekly data
    download_export(year=2023, month=1, week=1)