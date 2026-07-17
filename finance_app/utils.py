import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def setup_logging(log_path: str) -> None:
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        encoding="utf-8",
    )


def normalize_col(column: object) -> str:
    text = str(column).strip().lower().replace("\ufeff", "")
    text = re.sub(r"\s+", "_", text)
    text = text.replace("adjclose", "adj_close")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text


def safe_to_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="ISO8601")
    except TypeError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        parsed = pd.to_datetime(series, errors="coerce", format=fmt)
        if len(parsed) and float(parsed.notna().mean()) >= 0.75:
            return parsed
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def guess_company_from_path(path: str) -> str:
    name = Path(path).name.lower()
    return name.split("_")[0].replace(".csv", "") or "company"


def find_matching_column(columns: Iterable[str], names: Iterable[str]) -> Optional[str]:
    lowered = {str(col).lower(): str(col) for col in columns}
    for name in names:
        needle = name.lower()
        for lower, original in lowered.items():
            if needle in lower:
                return original
    return None

