# 📈 Stock Price Forecasting with LSTM

> *"Nobody can predict the market. But a model can learn its shape well enough to be a useful — and very humble — guide."*

My first time-series deep learning project. An LSTM in PyTorch trained on **1,255 real daily trading sessions of AAPL** pulled live from Yahoo Finance. The honest headline: this model tracks trend well and lags at turning points, and understanding *why* was worth more than any accuracy number.

## What this project does

- Downloads real daily OHLCV data for a ticker (AAPL by default) from Yahoo Finance, with retries.
- Caches the data locally so reruns work offline.
- Builds sliding 30-session windows and min-max normalises the series.
- Trains a 2-layer LSTM with dropout to predict the next session's close.
- Reports RMSE in dollars and MAPE.
- Forecasts 30 sessions ahead and charts predictions against actuals.

## The data

Yahoo Finance chart API, 5 years of daily bars:

| Column | Description |
|---|---|
| `date` | Trading session date |
| `open` / `high` / `low` / `close` | Daily OHLC prices |
| `volume` | Share volume |
| `ticker` | Symbol |

Retrieved range at time of writing: **2021-10-07 → 2026-10-07** (1,255 sessions).

## How to run it

```bash
pip install -r requirements.txt

python -m src.train_lstm   # fetches data, trains, forecasts
```

To pull fresh data instead of using the cache, delete `data/stock.csv` first. Swap the ticker by changing `TICKER` in `src/data_gen.py`.

## Project structure

```
stock-price-forecasting-lstm/
├── src/
│   ├── data_gen.py     # Yahoo Finance fetch + cache
│   ├── model.py        # PriceLSTM definition
│   └── train_lstm.py   # sequences, training, forecast, chart
├── data/
│   └── stock.csv
├── plots/
│   └── forecast.png
├── tests/
├── requirements.txt
└── README.md
```

## What I learned

- Why time series needs sequence models: the order of observations *is* the signal.
- That you must never feed raw dollar values to a network — normalise first.
- How a sequence model learns momentum and quietly fails at reversals.
- That MAPE is the metric that made sense here, because RMSE in dollars looks precise and isn't.

## Results

80/20 chronological split, 40 epochs, 30-session windows:

| Metric | Value |
|---|---|
| Test RMSE | $11.01 |
| Test MAPE | 3.03% |
| Training MSE | 0.0009 (normalised units) |

A 3% average error is a reasonable trend-follower, not an oracle. The forecast chart marks the actual/predicted boundary with a dotted line — past it, the model is extrapolating, and you can see it doesn't know what happens next.

---

*Built with Python, PyTorch, pandas, matplotlib. Real market data, honestly measured. Not financial advice.*
