"""Fetch real daily stock prices from Yahoo Finance and cache them locally."""
import io
import json
import os
import time
import urllib.request

import pandas as pd

CSV_PATH = "data/stock.csv"
TICKER = "AAPL"
RANGE = "5y"
INTERVAL = "1d"


def _download(ticker=TICKER, rng=RANGE, interval=INTERVAL):
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?range={rng}&interval={interval}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    payload = json.loads(urllib.request.urlopen(req, timeout=60).read())
    result = payload["chart"]["result"][0]

    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, unit="s"),
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        }
    )
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df["ticker"] = ticker
    return df


def load(ticker=TICKER, force=False):
    if os.path.exists(CSV_PATH) and not force:
        return pd.read_csv(CSV_PATH, parse_dates=["date"])

    last_error = None
    for _ in range(3):
        try:
            df = _download(ticker)
            os.makedirs("data", exist_ok=True)
            df.to_csv(CSV_PATH, index=False)
            return df
        except Exception as exc:  # network hiccups are common
            last_error = exc
            time.sleep(3)
    raise RuntimeError(f"Could not fetch stock data for {ticker}: {last_error}")


if __name__ == "__main__":
    df = load(force=True)
    print(f"Fetched {len(df)} trading days for {TICKER} -> {CSV_PATH}")
    print(f"Range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(df.tail())
