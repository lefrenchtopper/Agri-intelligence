from pathlib import Path
import shutil
import time

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager


# ============================================================
# CONFIG
# ============================================================

FIREFOX_PATH = r"C:\Program Files\Mozilla Firefox\firefox.exe"

YEAR = 2026
START_MONTH = 1
END_MONTH = 8

# Agmarknet commonly uses weeks 1-4.
# We will process them strictly in order.
MAX_WEEKS = 4

DOWNLOAD_ROOT = Path("data/raw/agmarknet") / str(YEAR)
TEMP_DOWNLOAD_FOLDER = DOWNLOAD_ROOT / ".downloads"

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
]

USABLE_EXTENSIONS = {".xlsx", ".xls", ".csv"}

DOWNLOAD_TIMEOUT_SECONDS = 45

TEMP_DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

BASE = (
    "https://api.agmarknet.gov.in/v1/price-trend/"
    "wholesale-prices-weekly"
)


# ============================================================
# FIREFOX SETUP
# ============================================================

options = webdriver.FirefoxOptions()
options.binary_location = FIREFOX_PATH

options.set_preference(
    "browser.download.folderList",
    2
)

options.set_preference(
    "browser.download.dir",
    str(TEMP_DOWNLOAD_FOLDER.resolve())
)

options.set_preference(
    "browser.download.useDownloadDir",
    True
)

options.set_preference(
    "browser.helperApps.neverAsk.saveToDisk",
    ",".join([
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ])
)

options.set_preference(
    "browser.download.manager.showWhenStarting",
    False
)

options.set_preference(
    "browser.download.alwaysOpenPanel",
    False
)


# ============================================================
# CLEAN TEMPORARY DOWNLOAD FOLDER
# ============================================================

for file in TEMP_DOWNLOAD_FOLDER.iterdir():
    if file.is_file():
        try:
            file.unlink()
        except PermissionError:
            pass


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def get_files():
    """Return all files currently in the temporary folder."""
    return {
        path
        for path in TEMP_DOWNLOAD_FOLDER.iterdir()
        if path.is_file()
    }


def wait_for_download(previous_files):
    """
    Wait until Firefox creates a new usable Excel/CSV file.

    Also waits for Firefox's temporary .part file to disappear,
    meaning the download has finished.
    """

    deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SECONDS

    while time.monotonic() < deadline:

        current_files = get_files()

        # Firefox uses .part while downloading.
        partial_files = [
            path
            for path in current_files
            if path.suffix.lower() == ".part"
        ]

        # Look for newly-created usable files.
        candidates = [
            path
            for path in current_files - previous_files
            if path.suffix.lower() in USABLE_EXTENSIONS
            and path.stat().st_size > 0
        ]

        if candidates and not partial_files:
            return max(
                candidates,
                key=lambda path: path.stat().st_mtime
            )

        time.sleep(1)

    return None


def download_week(driver, month, week):

    month_name = MONTH_NAMES[month - 1]

    target_folder = (
        DOWNLOAD_ROOT
        / month_name
        / f"Week_{week}"
    )

    target_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    url = (
        f"{BASE}"
        f"?report_mode=Districtwise"
        f"&commodity=23"
        f"&year={YEAR}"
        f"&month={month}"
        f"&week={week}"
        f"&state=31"
        f"&district=0"
        f"&export=true"
    )

    print()
    print("=" * 70)
    print(f"Downloading: {YEAR}-{month:02d} Week {week}")
    print(f"URL: {url}")
    print(f"Destination: {target_folder}")

    previous_files = get_files()

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT use driver.get().
    #
    # Selenium can wait indefinitely for this API download.
    # execute_script changes the browser location without
    # making Selenium wait for the navigation to complete.
    # --------------------------------------------------------

    driver.execute_script(
        "window.location.href = arguments[0];",
        url
    )

    print("Waiting for Excel download...")

    report = wait_for_download(previous_files)

    if report is None:
        print(
            f"FAILED: No download received within "
            f"{DOWNLOAD_TIMEOUT_SECONDS} seconds."
        )
        return False

    destination = target_folder / report.name

    if destination.exists():
        destination.unlink()

    shutil.move(
        str(report),
        str(destination)
    )

    print(f"SUCCESS: {destination}")

    return True


# ============================================================
# START FIREFOX
# ============================================================

print("=" * 70)
print("AGMARKNET WEEKLY PRICE DOWNLOADER")
print("=" * 70)
print(f"Year: {YEAR}")
print(f"Months: {MONTH_NAMES[START_MONTH - 1]} through {MONTH_NAMES[END_MONTH - 1]}")
print(f"Commodity: Onion (23)")
print(f"State: Tamil Nadu (31)")
print("=" * 70)

driver = webdriver.Firefox(
    service=Service(
        GeckoDriverManager().install()
    ),
    options=options,
)


# ============================================================
# DOWNLOAD IN STRICT CHRONOLOGICAL ORDER
# ============================================================

successful = 0
failed = 0

try:

    for month in range(
        START_MONTH,
        END_MONTH + 1
    ):

        print()
        print("#" * 70)
        print(
            f"STARTING MONTH: "
            f"{MONTH_NAMES[month - 1]}"
        )
        print("#" * 70)

        for week in range(
            1,
            MAX_WEEKS + 1
        ):

            success = download_week(
                driver,
                month,
                week
            )

            if success:
                successful += 1
            else:
                failed += 1

            # Small pause between requests.
            time.sleep(2)

finally:

    driver.quit()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)
print(f"Successful: {successful}")
print(f"Failed:     {failed}")
print(f"Output:     {DOWNLOAD_ROOT.resolve()}")
print("=" * 70)