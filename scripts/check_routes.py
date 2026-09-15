import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.main import app

print("\nRegistered Routes in FastAPI:")


def print_routes(routes):
    for route in routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            methods = f"[{', '.join(route.methods)}]"
            print(f"  {methods:<15} {route.path}")
        elif hasattr(route, "routes"):
            print_routes(route.routes)


print_routes(app.routes)