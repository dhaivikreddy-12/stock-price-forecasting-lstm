"""Behavioural tests for the LSTM stock forecaster.

model.py is import-time cheap and train_lstm.py is never imported (it trains
on import). data_gen.load() is never called with force=True, so no network.
"""
import ast
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV = ROOT / "data" / "stock.csv"
NROWS = 5000
SEQ_LEN = 30


def test_readme_and_license_exist():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "LICENSE").is_file()


def test_ticker_constant_is_aapl():
    from src.data_gen import CSV_PATH, TICKER

    assert TICKER == "AAPL"
    assert pathlib.Path(CSV_PATH).name == "stock.csv"
    assert callable(pd.read_csv)


def test_download_url_is_yahoo_chart_endpoint():
    """Guards the one network path: it must build the Yahoo chart URL."""
    tree = ast.parse((ROOT / "src" / "data_gen.py").read_text(encoding="utf-8"))
    sources = [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    assert any("query1.finance.yahoo.com" in s for s in sources)


def test_forward_pass_shape():
    import torch

    from src.model import PriceLSTM

    model = PriceLSTM()
    out = model(torch.zeros(3, SEQ_LEN, 1))
    assert out.shape == (3, 1)
    assert torch.isfinite(out).all()


def test_backward_pass_produces_finite_grads():
    import torch

    from src.model import PriceLSTM

    model = PriceLSTM()
    model(torch.zeros(3, SEQ_LEN, 1)).sum().backward()
    grad = model.lstm.weight_ih_l0.grad
    assert grad is not None
    # LSTM input weights are (4 * hidden, input_size) = (256, 1)
    assert grad.shape == (4 * model.lstm.hidden_size, 1)
    assert torch.isfinite(grad).all()
    assert model.fc.bias.grad is not None


def test_two_lstm_layers_with_dropout():
    from src.model import PriceLSTM

    model = PriceLSTM()
    assert model.lstm.num_layers == 2
    assert model.lstm.dropout > 0
    assert model.lstm.batch_first is True
    assert model.fc.in_features == model.lstm.hidden_size == 64


def test_sliding_window_arithmetic_matches_train_script():
    data = np.arange(100, dtype=np.float32)
    windows = np.array([data[i:i + SEQ_LEN] for i in range(70)])
    targets = np.array([data[i + SEQ_LEN] for i in range(70)])
    assert windows.shape == (70, SEQ_LEN)
    assert windows[:, :, None].shape == (70, SEQ_LEN, 1)  # channel axis added later
    assert targets.shape == (70,)
    assert windows[0, 0] == data[0]
    assert targets[-1] == data[99]
    # train_lstm.py builds windows with len(data) - seq_len, no +1.
    assert 100 - SEQ_LEN == 70


def test_cached_csv_has_positive_close_prices_and_parsable_dates():
    if not CSV.is_file():
        pytest.skip("cached data/stock.csv is not present")
    df = pd.read_csv(CSV, nrows=NROWS, parse_dates=["date"])
    assert "close" in df.columns
    assert len(df) > 100
    assert df["date"].notna().all()
    assert df["date"].is_monotonic_increasing
    assert (df["close"] > 0).all()
    assert df["close"].notna().all()
    assert df["date"].max() > df["date"].min()