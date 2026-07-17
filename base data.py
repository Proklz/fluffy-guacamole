import contextlib
import io
import logging
import os
import time

import pandas as pd
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

BASE_DIR = "finance_bigdata_project"
START_DATE = "2010-01-01"
END_DATE = None
RETRY_COUNT = 3
RETRY_DELAY_SECONDS = 3


INSTRUMENTS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "google": "GOOGL",
    "meta": "META",
    "netflix": "NFLX",
    "intel": "INTC",
    "amd": "AMD",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "paypal": "PYPL",
    "coca_cola": "KO",
    "jpmorgan": "JPM",
    "bank_of_america": "BAC",
    "visa": "V",
    "mastercard": "MA",
    "walmart": "WMT",
    "mcdonalds": "MCD",
    "pepsico": "PEP",
    "exxon_mobil": "XOM",
    "chevron": "CVX",
    "boeing": "BA",
    "disney": "DIS",
    "uber": "UBER",
    "shopify": "SHOP",
    "broadcom": "AVGO",
    "qualcomm": "QCOM",
    "ibm": "IBM",
    "eli_lilly": "LLY",
    "unitedhealth": "UNH",
    "sp500": "^GSPC",
    "nasdaq100": "^NDX",
    "dow_jones": "^DJI",
    "russell2000": "^RUT",
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
}


def ensure_project_dirs() -> None:
    os.makedirs(os.path.join(BASE_DIR, "stocks"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "financial_statements"), exist_ok=True)


def download_ticker_history(ticker: str) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            data = yf_download_silent(ticker, period="max")
            if not data.empty:
                return filter_history_period(data)

            data = yf_download_silent(ticker, period="10y")
            if not data.empty:
                return filter_history_period(data)
        except Exception as exc:
            last_error = exc

        if attempt < RETRY_COUNT:
            print(f"  Попытка {attempt} не дала данных, повтор через {RETRY_DELAY_SECONDS} сек.")
            time.sleep(RETRY_DELAY_SECONDS)

    if last_error:
        print(f"  Ошибка yfinance: {last_error}")
    return pd.DataFrame()


def yf_download_silent(ticker: str, period: str) -> pd.DataFrame:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        return yf.download(
            ticker,
            period=period,
            auto_adjust=False,
            progress=False,
            threads=False,
        )


def filter_history_period(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result[result.index.notna()]
    result = result.loc[result.index >= pd.Timestamp(START_DATE)]
    if END_DATE:
        result = result.loc[result.index <= pd.Timestamp(END_DATE)]
    return result


def download_stock_data() -> None:
    print("Скачивание рыночных данных Yahoo Finance\n")
    for name, ticker in INSTRUMENTS.items():
        print(f"Скачивание котировок: {name} ({ticker})")
        data = download_ticker_history(ticker)
        if data.empty:
            print(
                f"  Нет данных: {name} ({ticker}). "
                "Это может быть временная ошибка Yahoo Finance, rate limit, проблема сети или VPN/прокси."
            )
            continue
        file_path = os.path.join(BASE_DIR, "stocks", f"{name}_stock_data.csv")
        data.to_csv(file_path)
        print(f"  Сохранено: {file_path}; строк: {len(data):,}")


def download_financial_statements() -> None:
    print("\nСкачивание финансовой отчетности\n")
    for name, ticker in INSTRUMENTS.items():
        if ticker.startswith("^") or ticker.endswith("-USD"):
            print(f"Пропуск отчетности для индекса/криптовалюты: {name} ({ticker})")
            continue

        print(f"Обработка отчетности: {name} ({ticker})")
        try:
            company = yf.Ticker(ticker)
            frames = []
            for statement_type, frame in {
                "financials": company.financials,
                "balance_sheet": company.balance_sheet,
                "cashflow": company.cashflow,
            }.items():
                if frame is None or frame.empty:
                    continue
                prepared = frame.copy()
                prepared["statement_type"] = statement_type
                prepared.reset_index(inplace=True)
                frames.append(prepared)
        except Exception as exc:
            print(f"  Ошибка получения отчетности: {exc}")
            continue

        if not frames:
            print(f"  Нет отчетности: {name} ({ticker})")
            continue

        combined = pd.concat(frames, ignore_index=True)
        file_path = os.path.join(BASE_DIR, "financial_statements", f"{name}_financial_data.csv")
        combined.to_csv(file_path, index=False)
        print(f"  Сохранено: {file_path}; строк: {len(combined):,}")


def create_combined_stock_dataset() -> None:
    print("\nСоздание объединенного датасета котировок\n")
    frames = []
    for name, ticker in INSTRUMENTS.items():
        file_path = os.path.join(BASE_DIR, "stocks", f"{name}_stock_data.csv")
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path)
        df["Company"] = name
        df["Ticker"] = ticker
        frames.append(df)

    if not frames:
        print("Нет файлов котировок для объединения")
        return

    big_dataset = pd.concat(frames, ignore_index=True)
    output_path = os.path.join(BASE_DIR, "all_companies_stock_bigdata.csv")
    big_dataset.to_csv(output_path, index=False)
    print(f"Сохранен объединенный датасет: {output_path}")
    print(f"Всего строк: {len(big_dataset):,}")


def main() -> None:
    ensure_project_dirs()
    download_stock_data()
    download_financial_statements()
    create_combined_stock_dataset()
    print("\nЗагрузка данных завершена.")


if __name__ == "__main__":
    main()
