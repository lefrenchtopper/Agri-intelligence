import requests
import json


URL = "https://api.agmarknet.gov.in/v1/daily-price-arrival/filters"

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


response = requests.get(
    URL,
    headers=headers,
    timeout=30,
)

print("Status code:", response.status_code)

response.raise_for_status()

data = response.json()

print("\nAPI connection successful.")

# ---------------------------------------------------------
# Find and display only relevant records
# ---------------------------------------------------------

def find_matches(obj, term, path="root"):
    matches = []

    if isinstance(obj, dict):
        for key, value in obj.items():

            if isinstance(value, str):
                if term.lower() in value.lower():
                    matches.append({
                        "path": f"{path}.{key}",
                        "record": obj
                    })

            elif isinstance(value, (dict, list)):
                matches.extend(
                    find_matches(
                        value,
                        term,
                        f"{path}.{key}"
                    )
                )

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            matches.extend(
                find_matches(
                    value,
                    term,
                    f"{path}[{index}]"
                )
            )

    return matches


terms = [
    "Coimbatore",
    "Onion",
    "Coconut",
    "Arecanut",
]


for term in terms:

    print("\n" + "=" * 70)
    print(f"{term.upper()} MATCHES")
    print("=" * 70)

    matches = find_matches(data, term)

    # Remove duplicate records
    unique_records = []
    seen = set()

    for match in matches:

        record = match["record"]

        record_key = json.dumps(
            record,
            sort_keys=True,
            default=str
        )

        if record_key not in seen:
            seen.add(record_key)
            unique_records.append(match)

    if not unique_records:
        print("No matches found.")
        continue

    print(f"Found {len(unique_records)} matching records.\n")

    # Only show first 20 matches
    for match in unique_records[:20]:

        print("Path:")
        print(match["path"])

        print("Record:")
        print(json.dumps(
            match["record"],
            indent=2,
            ensure_ascii=False,
            default=str
        ))

        print("-" * 70)

    if len(unique_records) > 20:
        print(
            f"... {len(unique_records) - 20} additional matches hidden."
        )