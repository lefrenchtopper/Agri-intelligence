import requests
import json


URL = "https://api.agmarknet.gov.in/v1/daily-price-arrival"

headers = {
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://agmarknet.gov.in",
    "Referer": "https://agmarknet.gov.in/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
}


params = {
    "state": 31,
    "district": 530,
    "market": 277,
    "commodity": 23,
    "year": 2026,
    "month": 8,
}


response = requests.get(
    URL,
    headers=headers,
    params=params,
    timeout=30,
)


print("Status code:", response.status_code)

print("\nRequest URL:")
print(response.url)

response.raise_for_status()

data = response.json()

print("\nAPI request successful.")

print("\nTop-level structure:")
print(data.keys())

print("\nResponse:")
print(json.dumps(
    data,
    indent=2,
    ensure_ascii=False
))