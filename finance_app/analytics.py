from typing import List, Optional

import numpy as np
import pandas as pd

from .utils import find_matching_column


def dataset_overview(stock_df: pd.DataFrame, finance_df: pd.DataFrame) -> str:
    lines = ["=== ОБЗОР НАБОРА ФИНАНСОВЫХ ДАННЫХ ===", ""]
    stock_rows = len(stock_df)
    finance_rows = len(finance_df)
    lines.append(f"Строк котировок: {stock_rows:,}")
    lines.append(f"Строк финансовой отчетности: {finance_rows:,}")
    lines.append(f"Всего записей в аналитическом хранилище: {stock_rows + finance_rows:,}")

    if not stock_df.empty:
        stock_df = stock_df.copy()
        stock_df["date"] = pd.to_datetime(stock_df["date"], errors="coerce")
        companies = sorted(stock_df["company"].dropna().unique())
        lines.append(f"Компаний в котировках: {len(companies)} ({', '.join(companies)})")
        lines.append(f"Период котировок: {stock_df['date'].min().date()} - {stock_df['date'].max().date()}")
        checked_columns = [col for col in ["open", "high", "low", "close", "adj_close", "volume"] if col in stock_df.columns]
        missing = stock_df[checked_columns].isna().sum()
        lines.append("")
        lines.append("Пропуски по котировкам:")
        for col, count in missing.items():
            lines.append(f"- {col}: {int(count):,}")
        rows_by_company = stock_df.groupby("company").size().sort_values(ascending=False)
        lines.append("")
        lines.append("Строк по компаниям:")
        lines.append(rows_by_company.to_string())

    if not finance_df.empty:
        finance_companies = sorted(finance_df["company"].dropna().unique())
        lines.append("")
        lines.append(f"Компаний в отчетности: {len(finance_companies)} ({', '.join(finance_companies)})")
        lines.append("Типы отчетности:")
        lines.append(finance_df.groupby("statement_type").size().sort_values(ascending=False).to_string())

    lines.append("")
    lines.append("Комментарий для ВКР:")
    lines.append(
        "Приложение использует локальное аналитическое хранилище SQLite, пакетный импорт CSV и расчетные методы pandas/numpy "
        "для обработки финансовых временных рядов и отчетности."
    )
    return "\n".join(lines)


def company_summary(stock_df: pd.DataFrame) -> pd.DataFrame:
    if stock_df.empty:
        raise ValueError("Нет данных stock_prices")
    df = stock_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    rows = []
    for company, group in df.groupby("company"):
        group = group.sort_values("date")
        first = float(group["close"].iloc[0])
        last = float(group["close"].iloc[-1])
        daily_ret = group["close"].pct_change().dropna()
        total_return = (last / first - 1) * 100 if first else np.nan
        annual_vol = daily_ret.std() * np.sqrt(252) * 100 if len(daily_ret) else np.nan
        annual_return = daily_ret.mean() * 252 * 100 if len(daily_ret) else np.nan
        sharpe = annual_return / annual_vol if annual_vol and not np.isnan(annual_vol) else np.nan
        max_drawdown = ((group["close"] / group["close"].cummax()) - 1).min() * 100
        rows.append(
            [
                company,
                group["ticker"].iloc[-1],
                group["date"].min().date(),
                group["date"].max().date(),
                len(group),
                last,
                total_return,
                annual_return,
                annual_vol,
                sharpe,
                max_drawdown,
                group["volume"].mean(),
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "company",
            "ticker",
            "start",
            "end",
            "rows",
            "last_close",
            "total_return_%",
            "annual_return_%",
            "annual_volatility_%",
            "sharpe_ratio",
            "max_drawdown_%",
            "avg_volume",
        ],
    ).sort_values("total_return_%", ascending=False)


def technical_indicators(stock_df: pd.DataFrame) -> pd.DataFrame:
    if len(stock_df) < 60:
        raise ValueError("Нужно минимум 60 строк котировок")
    df = stock_df.copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df["daily_return"] = df["close"].pct_change()
    df["cum_return"] = (1 + df["daily_return"].fillna(0)).cumprod() - 1
    df["SMA20"] = df["close"].rolling(20).mean()
    df["SMA50"] = df["close"].rolling(50).mean()
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI14"] = 100 - (100 / (1 + rs))
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["Bollinger_upper"] = mid + 2 * std
    df["Bollinger_lower"] = mid - 2 * std
    return df


def technical_signal(indicators: pd.DataFrame) -> str:
    last = indicators.iloc[-1]
    signal = "HOLD"
    reasons = []
    if last["SMA20"] > last["SMA50"]:
        reasons.append("SMA20 выше SMA50: краткосрочный тренд сильнее долгосрочного")
        signal = "BUY"
    else:
        reasons.append("SMA20 ниже SMA50: тренд ослаблен")
    if last["RSI14"] > 70:
        reasons.append("RSI выше 70: возможная перекупленность")
        signal = "SELL / WAIT"
    elif last["RSI14"] < 30:
        reasons.append("RSI ниже 30: возможная перепроданность")
        signal = "BUY / WATCH"
    if last["MACD"] > last["MACD_signal"]:
        reasons.append("MACD выше сигнальной линии")
    else:
        reasons.append("MACD ниже сигнальной линии")
    out = [
        "=== ТЕХНИЧЕСКИЙ АНАЛИЗ ===",
        f"Дата: {last['date'].date()}",
        f"Цена закрытия: {last['close']:,.2f}",
        f"SMA20: {last['SMA20']:,.2f}",
        f"SMA50: {last['SMA50']:,.2f}",
        f"RSI14: {last['RSI14']:,.2f}",
        f"MACD: {last['MACD']:,.4f}",
        f"MACD signal: {last['MACD_signal']:,.4f}",
        f"Bollinger upper: {last['Bollinger_upper']:,.2f}",
        f"Bollinger lower: {last['Bollinger_lower']:,.2f}",
        "",
        f"Итоговый сигнал: {signal}",
        "Причины:",
    ]
    out.extend(f"- {reason}" for reason in reasons)
    out.extend(["", "Последние 15 строк:", indicators.tail(15).to_string(index=False)])
    return "\n".join(out)


def return_correlation(stock_df: pd.DataFrame) -> pd.DataFrame:
    if stock_df.empty:
        raise ValueError("Нет данных stock_prices")
    df = stock_df.copy()
    pivot = df.pivot_table(index="date", columns="company", values="close")
    returns = pivot.pct_change().dropna(how="all")
    return returns.corr()


def anomalous_days(stock_df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    df = stock_df.copy().sort_values(["company", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["daily_return"] = df.groupby("company")["close"].pct_change()
    stats = df.groupby("company")["daily_return"].agg(["mean", "std"])
    df = df.join(stats, on="company", rsuffix="_stat")
    df["z_score"] = (df["daily_return"] - df["mean"]) / df["std"]
    anomalies = df[df["z_score"].abs() >= threshold].copy()
    return anomalies[["company", "ticker", "date", "close", "daily_return", "z_score"]].sort_values("z_score", key=lambda s: s.abs(), ascending=False)


def financial_ratios(finance_df: pd.DataFrame) -> pd.DataFrame:
    if finance_df.empty:
        raise ValueError("Нет данных finance_statement по выбранной компании")
    pivot = finance_df.pivot_table(index="period", columns="item", values="value", aggfunc="first").sort_index()

    net_income = find_matching_column(pivot.columns, ["Net Income"])
    assets = find_matching_column(pivot.columns, ["Total Assets"])
    equity = find_matching_column(pivot.columns, ["Stockholders Equity", "Total Equity"])
    revenue = find_matching_column(pivot.columns, ["Total Revenue"])
    debt = find_matching_column(pivot.columns, ["Total Debt", "Long Term Debt"])
    fcf = find_matching_column(pivot.columns, ["Free Cash Flow"])

    result = pd.DataFrame(index=pivot.index)
    if revenue:
        result["Revenue"] = pivot[revenue]
    if net_income:
        result["Net Income"] = pivot[net_income]
    if fcf:
        result["Free Cash Flow"] = pivot[fcf]
    if net_income and revenue:
        result["Net Margin"] = pivot[net_income] / pivot[revenue]
    if net_income and assets:
        result["ROA"] = pivot[net_income] / pivot[assets]
    if net_income and equity:
        result["ROE"] = pivot[net_income] / pivot[equity]
    if debt and assets:
        result["Debt/Assets"] = pivot[debt] / pivot[assets]
    if revenue:
        result["Revenue Growth"] = pivot[revenue].pct_change()
    return result
