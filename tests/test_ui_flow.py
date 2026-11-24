"""Exercise the UI beyond the first render: build a plan and generate motion."""

import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / ".numba_cache"))

from streamlit.testing.v1 import AppTest


def main() -> int:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=900)
    at.run()

    if at.exception:
        print("initial render failed:", [e.value for e in at.exception])
        return 1

    # the dashboard is a hub; navigate to the plan workspace first
    at.session_state["nav"] = "Plan"
    at.run()
    buttons = {button.label: button for button in at.button}
    if "Build mapping plan" not in buttons:
        print("build button missing; buttons:", list(buttons))
        return 1
    buttons["Build mapping plan"].click()
    at.run()
    print("after build: exceptions =", len(at.exception))
    for error in at.exception:
        print(error.value)
    print("dataframes:", len(at.dataframe), "charts:", len(at.get("line_chart")))

    at.session_state["nav"] = "Motion"
    at.run()
    buttons = {button.label: button for button in at.button}
    if "Generate motion" not in buttons:
        print("generate button missing; buttons:", list(buttons))
        return 1
    buttons["Generate motion"].click()
    at.run()
    print("after generate: exceptions =", len(at.exception))
    for error in at.exception:
        print(error.value)
    print("metrics:", len(at.metric))
    return 0 if not at.exception else 1


if __name__ == "__main__":
    raise SystemExit(main())
