import os
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parent.parent

for env_path in (BACKEND_DIR / ".env", REPO_ROOT / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=True)


DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./agri_intelligence.db"