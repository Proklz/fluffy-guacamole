from dataclasses import dataclass, field
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import COMPANY_TICKERS
from .database import connect_db
from .utils import guess_company_from_path, normalize_col, safe_to_datetime

logger = logging.getLogger(__name__)


@dataclass
class ImportReport:
    source: str
    imported_rows: int = 0
    skipped_rows: int = 0
    files: int = 0
    errors: List[str] = field(default_factory=list)

    def merge(self, other: "ImportReport") -> None:
        self.imported_rows += other.imported_rows
        self.skipped_rows += other.skipped_rows
        self.files += other.files
        self.errors.extend(other.errors)

    def summary(self) -> str:
        lines = [
            f"Источник: {self.source}",
            f"Файлов обработано: {self.files}",
            f"Строк загружено/обновлено: {self.imported_rows}",
            f"Строк пропущено: {self.skipped_rows}",
        ]
        if self.errors:
            lines.append("Ошибки:")
            lines.extend(f"- {err}" for err in self.errors)
        return "\n".join(lines)


def import_stock_csv(db_path: str, path: str, chunksize: int = 50_000) -> ImportReport:
    report = ImportReport(source=path, files=1)
    reader = pd.read_csv(path, sep=None, engine="python", chunksize=chunksize)
    for chunk in reader:
        rows, skipped = _prepare_stock_rows(chunk, path)
        report.skipped_rows += skipped
        report.imported_rows += _upsert_stock_rows(db_path, rows)
    logger.info("Imported stock CSV %s: %s rows, %s skipped", path, report.imported_rows, report.skipped_rows)
    return report


def import_finance_csv(db_path: str, path: str) -> ImportReport:
    report = ImportReport(source=path, files=1)
    raw = pd.read_csv(path, sep=None, engine="python")
    rows, skipped = _prepare_finance_rows(raw, path)
    report.skipped_rows = skipped
    report.imported_rows = _upsert_finance_rows(db_path, rows)
    logger.info("Imported finance CSV %s: %s rows, %s skipped", path, report.imported_rows, report.skipped_rows)
    return report


def import_base_data_folder(db_path: str, folder: str, chunksize: int = 50_000) -> Tuple[ImportReport, ImportReport]:
    stock_report = ImportReport(source=os.path.join(folder, "stocks"))
    finance_report = ImportReport(source=os.path.join(folder, "financial_statements"))

    stock_dir = os.path.join(folder, "stocks")
    loaded_individual_stock_files = False
    if os.path.isdir(stock_dir):
        for name in sorted(os.listdir(stock_dir)):
            if name.lower().endswith(".csv"):
                try:
                    stock_report.merge(import_stock_csv(db_path, os.path.join(stock_dir, name), chunksize))
                    loaded_individual_stock_files = True
                except Exception as exc:
                    message = f"{name}: {exc}"
                    stock_report.errors.append(message)
                    logger.exception("Stock import failed: %s", message)

    big_file = os.path.join(folder, "all_companies_stock_bigdata.csv")
    if os.path.exists(big_file) and not loaded_individual_stock_files:
        try:
            stock_report.merge(import_stock_csv(db_path, big_file, chunksize))
        except Exception as exc:
            message = f"{os.path.basename(big_file)}: {exc}"
            stock_report.errors.append(message)
            logger.exception("Combined stock import failed: %s", message)

    fin_dir = os.path.join(folder, "financial_statements")
    if os.path.isdir(fin_dir):
        for name in sorted(os.listdir(fin_dir)):
            if name.lower().endswith(".csv"):
                try:
                    finance_report.merge(import_finance_csv(db_path, os.path.join(fin_dir, name)))
                except Exception as exc:
                    message = f"{name}: {exc}"
                    finance_report.errors.append(message)
                    logger.exception("Finance import failed: %s", message)

    return stock_report, finance_report


def _prepare_stock_rows(df: pd.DataFrame, path: str) -> Tuple[List[Tuple], int]:
    original_len = len(df)
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    cols = list(df.columns)

    date_col = next((c for c in ["date", "datetime", "timestamp", "time", "price"] if c in cols), None)
    if date_col is None:
        best_col, best_ratio = None, 0.0
        for col in cols:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                parsed = safe_to_datetime(df[col])
                ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
                if ratio > best_ratio:
                    best_col, best_ratio = col, ratio
        if best_col and best_ratio >= 0.75:
            date_col = best_col

    close_col = "close" if "close" in cols else ("adj_close" if "adj_close" in cols else None)
    if not date_col or not close_col:
        raise ValueError(f"Нужны колонки Date и Close. Найдены колонки: {cols}")

    guessed_company = guess_company_from_path(path)
    guessed_ticker = COMPANY_TICKERS.get(guessed_company, guessed_company.upper())
    company_col = "company" if "company" in cols else None
    ticker_col = "ticker" if "ticker" in cols else None

    def num(col: str) -> pd.Series:
        if col in cols:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series([np.nan] * len(df))

    out = pd.DataFrame(
        {
            "company": df[company_col].astype(str).str.lower() if company_col else guessed_company,
            "ticker": df[ticker_col].astype(str).str.upper() if ticker_col else guessed_ticker,
            "date": safe_to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
            "open": num("open"),
            "high": num("high"),
            "low": num("low"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
            "adj_close": num("adj_close"),
            "volume": num("volume"),
        }
    )
    out = out.dropna(subset=["date", "close"])
    skipped = original_len - len(out)
    rows = [tuple(None if pd.isna(value) else value for value in row) for row in out.itertuples(index=False, name=None)]
    if not rows:
        raise ValueError("После очистки CSV не осталось строк котировок")
    return rows, skipped


def _prepare_finance_rows(raw: pd.DataFrame, path: str) -> Tuple[List[Tuple], int]:
    original_cols = [str(c).strip() for c in raw.columns]
    norm_cols = [normalize_col(c) for c in original_cols]
    original_col_by_norm: Dict[str, str] = dict(zip(norm_cols, original_cols))
    raw = raw.copy()
    raw.columns = norm_cols

    company = guess_company_from_path(path)
    ticker = COMPANY_TICKERS.get(company, company.upper())
    long_rows: List[Tuple] = []
    skipped = 0

    if "statement_type" in raw.columns and ("index" in raw.columns or "item" in raw.columns):
        item_col = "item" if "item" in raw.columns else "index"
        value_cols = [c for c in raw.columns if c not in {item_col, "statement_type", "company", "ticker"}]
        for _, row in raw.iterrows():
            item = str(row[item_col]).strip()
            st = str(row["statement_type"]).strip()
            for period_col in value_cols:
                period_label = original_col_by_norm.get(period_col, period_col)
                period_dt = safe_to_datetime(pd.Series([period_label])).iloc[0]
                val = pd.to_numeric(pd.Series([row[period_col]]), errors="coerce").iloc[0]
                if pd.isna(period_dt) or pd.isna(val):
                    skipped += 1
                    continue
                long_rows.append((company, ticker, st, item, period_dt.strftime("%Y-%m-%d"), float(val)))
    elif {"company", "ticker", "statement_type", "item", "period", "value"}.issubset(set(raw.columns)):
        for _, row in raw.iterrows():
            period_dt = safe_to_datetime(pd.Series([row["period"]])).iloc[0]
            val = pd.to_numeric(pd.Series([row["value"]]), errors="coerce").iloc[0]
            if pd.isna(period_dt):
                skipped += 1
                continue
            long_rows.append(
                (
                    str(row["company"]).strip().lower(),
                    str(row["ticker"]).strip().upper(),
                    str(row["statement_type"]).strip(),
                    str(row["item"]).strip(),
                    period_dt.strftime("%Y-%m-%d"),
                    None if pd.isna(val) else float(val),
                )
            )
    else:
        raise ValueError("Не распознан формат отчетности. Нужен файл из financial_statements или long-формат.")

    if not long_rows:
        raise ValueError("Не найдено строк отчетности для импорта")
    return long_rows, skipped


def _upsert_stock_rows(db_path: str, rows: List[Tuple]) -> int:
    with connect_db(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO stock_prices(company,ticker,date,open,high,low,close,adj_close,volume)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(company,ticker,date) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                adj_close=excluded.adj_close,
                volume=excluded.volume
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def _upsert_finance_rows(db_path: str, rows: List[Tuple]) -> int:
    with connect_db(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO finance_statement(company,ticker,statement_type,item,period,value)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(company,ticker,statement_type,item,period) DO UPDATE SET
                value=excluded.value
            """,
            rows,
        )
        conn.commit()
    return len(rows)
