"""Generate a synthetic stock price time series (random walk with drift)."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

n_days = 750
start_price = 100.0
drift = 0.0004
vol = 0.018

returns = rng.normal(drift, vol, n_days)
prices = start_price * np.exp(np.cumsum(returns))

dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")

df = pd.DataFrame({"date": dates, "close": prices.round(2)})

os.makedirs("data", exist_ok=True)
df.to_csv("data/stock.csv", index=False)
print(f"Generated {len(df)} days of prices -> data/stock.csv")
print(df.head())
