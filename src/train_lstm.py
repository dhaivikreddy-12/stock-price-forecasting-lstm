"""Train an LSTM on real stock prices and forecast the next 30 sessions."""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import PriceLSTM
from src.data_gen import load, TICKER

torch.manual_seed(0)

df = load()
print(f"Loaded {len(df)} sessions for {TICKER}")
print(f"Range: {df['date'].min().date()} -> {df['date'].max().date()}")

prices = df["close"].to_numpy(dtype=np.float32)

# Normalise to [0, 1] - never feed raw dollar values to an LSTM.
pmin, pmax = float(prices.min()), float(prices.max())
norm = (prices - pmin) / (pmax - pmin)


def make_sequences(data, seq_len):
    xs = np.array([data[i:i + seq_len] for i in range(len(data) - seq_len)])
    ys = np.array([data[i + seq_len] for i in range(len(data) - seq_len)])
    return xs[:, :, None], ys


SEQ_LEN = 30
X, y = make_sequences(norm, SEQ_LEN)

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=256)

model = PriceLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 40
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb).squeeze(), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:>2}/{EPOCHS} | train MSE {running/len(X_train):.6f}")

model.eval()
with torch.no_grad():
    preds = torch.cat([model(xb) for xb, _ in test_loader]).squeeze().numpy()

pred_prices = preds * (pmax - pmin) + pmin
actual_prices = y_test * (pmax - pmin) + pmin

rmse = float(np.sqrt(np.mean((pred_prices - actual_prices) ** 2)))
mape = float(np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100)
print(f"\nTest RMSE ${rmse:.2f} | MAPE {mape:.2f}%")

# Forecast 30 sessions beyond the end of the data.
current = torch.tensor(X[-1:]).float()
future = []
with torch.no_grad():
    for _ in range(30):
        p = model(current)
        future.append(p.item())
        current = torch.cat([current[:, 1:, :], p.view(1, 1, 1)], dim=1)
future = np.array(future) * (pmax - pmin) + pmin

test_start = len(prices) - len(actual_prices)
plt.figure(figsize=(12, 5))
plt.plot(range(test_start, test_start + len(actual_prices)), actual_prices, label="Actual", alpha=0.8)
plt.plot(range(test_start, test_start + len(pred_prices)), pred_prices, label="LSTM prediction", alpha=0.8)
plt.plot(range(len(prices), len(prices) + 30), future, "k--", label="Forecast +30 sessions")
plt.axvline(len(prices), color="grey", lw=0.8, ls=":")
plt.xlabel("Trading session")
plt.ylabel("Close price ($)")
plt.title(f"{TICKER} LSTM Forecast (RMSE ${rmse:.2f})")
plt.legend()
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/forecast.png", dpi=120)
print("Saved forecast chart to plots/forecast.png")
print(f"Last close ${prices[-1]:.2f} -> 30-session forecast ${future[-1]:.2f}")
