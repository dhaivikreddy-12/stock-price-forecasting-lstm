# 📈 Stock Price Forecasting with LSTM

> *"Nobody can predict the market. But a model can learn its patterns well enough to be a useful (and humble) guide."*

My first time-series deep learning project. It uses an LSTM (Long Short-Term Memory) network in PyTorch to forecast stock prices from historical data. I learned quickly that predicting prices is *hard* — the goal here isn't to get rich, it's to understand how sequence models work and to be honest about their limits.

## What this project does

- Generates (or loads) a realistic stock-price time series.
- Builds sliding-window sequences for the LSTM to learn from.
- Defines a small LSTM model in PyTorch.
- Trains it to predict the next day's price from the past 30 days.
- Compares predictions against actual prices with a clean chart.

## The data

I generated synthetic-but-plausible daily prices (a random-walk with drift and volatility), so the project runs offline. The repo also supports swapping in real data via `data/stock.csv` if you have any.

## How to run it

```bash
pip install -r requirements.txt

# Generate data + train + forecast
python train_lstm.py
```

## Project structure

```
stock-price-forecasting-lstm/
├── src/
│   ├── data_gen.py        # synthetic price generation
│   ├── model.py           # LSTM definition
│   └── train_lstm.py      # training + forecasting
├── plots/
│   └── forecast.png       # actual vs predicted chart
├── requirements.txt
└── README.md
```

## What I learned

- Why time series needs sequence models (order matters!).
- What an LSTM's memory cells actually help with — and don't.
- The danger of overfitting a chart that "looks right".
- That you must normalize prices, never feed raw dollars to a network.

## Results

The LSTM learns the *trend* well and tracks short-term moves reasonably, but — as expected — it lags at turning points. That's the honest reality of financial forecasting, and it's the most valuable lesson this project teaches.

---

*Built with Python, PyTorch, pandas, matplotlib. Made for learning, by a student, for students.*
