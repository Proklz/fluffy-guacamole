from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AppConfig:
    db_path: str = "finance_bigdata.db"
    theme: str = "superhero"
    log_path: str = "app.log"
    csv_chunksize: int = 50_000


CONFIG = AppConfig()


COMPANY_TICKERS: Dict[str, str] = {
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
