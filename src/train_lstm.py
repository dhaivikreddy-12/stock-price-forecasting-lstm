"""Train LSTM and forecast future stock prices."""
import os
import subprocess
import sys

if not os.path.exists("data/stock.csv"):
    subprocess.run([sys.executable, "src/data_gen.py"], check=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.model import PriceLSTM

torch.manual_seed(0)

df = pd.read_csv("data/stock.csv")
prices = df["close"].values.astype(np.float32)

# Normalize to [0, 1]
pmin, pmax = prices.min(), prices.max()
prices_n = (prices - pmin) / (pmax - pmin)


def make_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len])
        ys.append(data[i + seq_len])
    return np.array(xs), np.array(ys)


SEQ_LEN = 30
X, y = make_sequences(prices_n, SEQ_LEN)
X = X[:, :, None]

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128)

model = PriceLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 25
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out.squeeze(), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:>2}/{epochs} | loss {total_loss/len(train_ds):.6f}")

model.eval()
with torch.no_grad():
    preds = model(torch.tensor(X_test)).squeeze().numpy()

preds_prices = preds * (pmax - pmin) + pmin
actual_prices = y_test * (pmax - pmin) + pmin

# Forecast next 30 days beyond the data
last_seq = torch.tensor(X[-1:]).float()
future = []
current = last_seq
with torch.no_grad():
    for _ in range(30):
        p = model(current)
        future.append(p.item())
        current = torch.cat([current[:, 1:, :], p.view(1, 1, 1)], dim=1)
future = np.array(future) * (pmax - pmin) + pmin

plt.figure(figsize=(12, 5))
plt.plot(range(len(prices_n) - len(X_test), len(prices_n) - len(X_test) + len(actual_prices)),
         actual_prices, label="Actual", alpha=0.7)
plt.plot(range(len(prices_n) - len(X_test), len(prices_n) - len(X_test) + len(preds_prices)),
         preds_prices, label="Predicted (test)", alpha=0.7)
plt.plot(range(len(prices_n), len(prices_n) + 30), future, label="Forecast +30 days", linestyle="--")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.title("LSTM Stock Price Forecast")
plt.legend()
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/forecast.png", dpi=120)
print("Saved forecast chart to plots/forecast.png")
print(f"Final predicted price: ${future[-1]:.2f}")
