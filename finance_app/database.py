import sqlite3
from typing import Iterable, Sequence, Tuple

import pandas as pd


STOCK_COLUMNS = ["id", "company", "ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
FINANCE_COLUMNS = ["id", "company", "ticker", "statement_type", "item", "period", "value"]


def connect_db(db_path: str, timeout: int = 30) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db(db_path: str) -> None:
    with connect_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL NOT NULL,
                adj_close REAL,
                volume REAL,
                UNIQUE(company, ticker, date)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS finance_statement (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                ticker TEXT NOT NULL,
                statement_type TEXT NOT NULL,
                item TEXT NOT NULL,
                period TEXT NOT NULL,
                value REAL,
                UNIQUE(company, ticker, statement_type, item, period)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_stock_main ON stock_prices(company, ticker, date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_prices(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fin_main ON finance_statement(company, ticker, statement_type, item, period)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fin_period ON finance_statement(period)")
        conn.commit()


def query_df(db_path: str, sql: str, params: Tuple = ()) -> pd.DataFrame:
    with connect_db(db_path) as conn:
        return pd.read_sql(sql, conn, params=params)


def execute(db_path: str, sql: str, params: Tuple = ()) -> int:
    with connect_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        return cur.rowcount


def clear_database(db_path: str) -> None:
    with connect_db(db_path) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM finance_statement")
        cur.execute("DELETE FROM stock_prices")
        cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('finance_statement', 'stock_prices')")
        conn.commit()


def executemany(db_path: str, sql: str, rows: Iterable[Sequence]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with connect_db(db_path) as conn:
        conn.executemany(sql, rows)
        conn.commit()
    return len(rows)
