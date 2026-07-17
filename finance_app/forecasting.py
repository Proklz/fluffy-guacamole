from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.neural_network import MLPRegressor  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    HAS_SKLEARN = True
except Exception:
    RandomForestRegressor = None  # type: ignore
    MLPRegressor = None  # type: ignore
    StandardScaler = None  # type: ignore
    HAS_SKLEARN = False


ALGORITHMS = ["linear", "moving_average", "random_forest", "neural_network_mlp"]


def forecast_series(series: pd.Series, steps: int, algorithm: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float).values
    if len(values) < 2:
        raise ValueError("Для прогноза нужно минимум 2 числовые точки")
    if algorithm == "linear":
        x = np.arange(len(values), dtype=float)
        a, b = np.polyfit(x, values, 1)
        return a * np.arange(len(values), len(values) + steps) + b
    if algorithm == "moving_average":
        window = min(30, len(values))
        return np.array([np.mean(values[-window:])] * steps)
    if algorithm in {"random_forest", "neural_network_mlp"}:
        if not HAS_SKLEARN:
            raise ValueError("Для этого алгоритма установите scikit-learn: pip install scikit-learn")
        lookback = min(30, max(2, len(values) // 4))
        X, y = [], []
        for i in range(lookback, len(values)):
            X.append(values[i - lookback : i])
            y.append(values[i])
        X = np.array(X)
        y = np.array(y)
        if algorithm == "random_forest":
            model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2)  # type: ignore
            model.fit(X, y)
            return _recursive_predict(model, values[-lookback:], steps)
        scaler_x = StandardScaler()  # type: ignore
        scaler_y = StandardScaler()  # type: ignore
        Xs = scaler_x.fit_transform(X)
        ys = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=700, random_state=42)  # type: ignore
        model.fit(Xs, ys)
        last = values[-lookback:].copy()
        preds = []
        for _ in range(steps):
            scaled = model.predict(scaler_x.transform(last.reshape(1, -1)))[0]
            pred = float(scaler_y.inverse_transform([[scaled]])[0][0])
            preds.append(pred)
            last = np.roll(last, -1)
            last[-1] = pred
        return np.array(preds)
    raise ValueError("Неизвестный алгоритм")


def backtest_series(series: pd.Series, algorithms: List[str], test_size: int = 60) -> pd.DataFrame:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float).reset_index(drop=True)
    if len(values) < test_size + 80:
        test_size = max(10, min(30, len(values) // 4))
    if len(values) <= test_size + 2:
        raise ValueError("Недостаточно данных для backtesting")
    train = values.iloc[:-test_size]
    test = values.iloc[-test_size:].values
    rows: List[Dict[str, float | str]] = []
    for algorithm in algorithms:
        try:
            pred = forecast_series(train, test_size, algorithm)
            error = pred - test
            mae = float(np.mean(np.abs(error)))
            rmse = float(np.sqrt(np.mean(error**2)))
            denom = np.where(test == 0, np.nan, test)
            mape = float(np.nanmean(np.abs(error / denom)) * 100)
            rows.append({"algorithm": algorithm, "MAE": mae, "RMSE": rmse, "MAPE_%": mape})
        except Exception as exc:
            rows.append({"algorithm": algorithm, "MAE": np.nan, "RMSE": np.nan, "MAPE_%": np.nan, "error": str(exc)})
    return pd.DataFrame(rows).sort_values("RMSE", na_position="last")


def finance_forecast(finance_df: pd.DataFrame, periods: int) -> pd.DataFrame:
    if len(finance_df) < 2:
        raise ValueError("Нужно минимум 2 точки отчетности")
    df = finance_df.copy()
    df["period_dt"] = pd.to_datetime(df["period"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["period_dt", "value"]).sort_values("period_dt")
    if len(df) < 2:
        raise ValueError("После очистки осталось меньше 2 точек отчетности")
    y = df["value"].astype(float).values
    x = np.arange(len(y), dtype=float)
    a, b = np.polyfit(x, y, 1)
    pred = a * np.arange(len(y), len(y) + periods) + b
    diffs = df["period_dt"].diff().dropna().dt.days.values
    step_days = int(np.median(diffs)) if len(diffs) else 365
    if step_days <= 0:
        step_days = 365
    dates = [df["period_dt"].max() + pd.Timedelta(days=step_days * (i + 1)) for i in range(periods)]
    return pd.DataFrame({"period": [d.date().isoformat() for d in dates], "forecast": pred})


def _recursive_predict(model, last_values: np.ndarray, steps: int) -> np.ndarray:
    last = last_values.copy()
    preds = []
    for _ in range(steps):
        pred = float(model.predict(last.reshape(1, -1))[0])
        preds.append(pred)
        last = np.roll(last, -1)
        last[-1] = pred
    return np.array(preds)

