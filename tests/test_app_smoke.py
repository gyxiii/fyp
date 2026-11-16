"""Headless smoke test of the Streamlit interface using streamlit.testing."""

import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / ".numba_cache"))

from streamlit.testing.v1 import AppTest


def main() -> int:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=600)
    app.run()

    errors = list(app.exception)
    if errors:
        print(f"FAILED: {len(errors)} exception(s)")
        for item in errors:
            print("---")
            print(item.value)
        return 1

    print("App rendered without exceptions")
    print(f"tabs: {[tab.label for tab in app.tabs]}")
    print(f"buttons: {[button.label for button in app.button]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
