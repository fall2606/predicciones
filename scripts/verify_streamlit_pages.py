"""Run each Streamlit page in AppTest and verify real CSV reads without editing the ledgers."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
# Streamlit runs page scripts with the project root on sys.path; mirror that in AppTest.
sys.path.insert(0, str(ROOT))
PAGES = [
    ("MLB", ROOT / "pages" / "MLB.py", ("predicciones_mlb.csv", "resultados_mlb.csv")),
    ("LaLiga", ROOT / "pages" / "LaLiga.py", ("predicciones_laliga.csv", "resultados_laliga.csv")),
    ("Champions", ROOT / "pages" / "Champions.py", ("predicciones_champions.csv", "resultados_champions.csv")),
    ("NFL", ROOT / "pages" / "NFL.py", ("predicciones_nfl.csv", "resultados_nfl.csv")),
]
original_read_csv = pd.read_csv
reads = []


def tracked_read_csv(filepath_or_buffer, *args, **kwargs):
    try:
        name = Path(str(filepath_or_buffer)).name
        if name.endswith(".csv"):
            reads.append(name)
    except Exception:
        pass
    return original_read_csv(filepath_or_buffer, *args, **kwargs)


pd.read_csv = tracked_read_csv
os.environ["NFL_PAGE_TEST_MODE"] = "1"
try:
    for sport, path, required_csvs in PAGES:
        app = AppTest.from_file(str(path), default_timeout=90).run()
        errors = list(app.exception)
        if errors:
            details = "; ".join(str(item.message) for item in errors)
            raise AssertionError(f"{sport}: Streamlit AppTest produjo excepción: {details}")
        missing = [name for name in required_csvs if not Path(ROOT / name).exists()]
        if missing:
            raise AssertionError(f"{sport}: faltan CSV reales: {missing}")
        loaded = [name for name in required_csvs if name in reads]
        if sport == "NFL":
            # NFL's tolerant reader uses csv.reader; its footer also reads the actual ledgers with pandas.
            loaded = [name for name in required_csvs if name in reads]
        if len(loaded) != len(required_csvs):
            raise AssertionError(
                f"{sport}: la página no leyó sus CSV reales {required_csvs}; lecturas observadas={loaded}"
            )
        print(f"PAGE_LOAD_OK {sport}: {path.relative_to(ROOT)}; CSV reales leídos={','.join(loaded)}; exceptions=0")
finally:
    pd.read_csv = original_read_csv
