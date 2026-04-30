from __future__ import annotations

"""
Experiment 3: RNN, LSTM, and GRU for time-series forecasting.

The task is sequence-to-one regression:
Given a window of previous time-series values, predict the next value.
"""

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path("outputs") / "03_timeseries_rnn"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_time_series(n_points: int = 2400) -> np.ndarray:
    t = np.arange(n_points, dtype=float)

    signal = (
        0.70 * np.sin(2 * np.pi * t / 40)
        + 0.35 * np.sin(2 * np.pi * t / 100)
        + 0.20 * np.cos(2 * np.pi * t / 17)
        + 0.0008 * t
    )

    nonlinear = 0.18 * np.sin(signal * 2.5)
    noise = np.random.default_rng(SEED).normal(0.0, 0.08, size=n_points)

    return signal + nonlinear + noise


def make_supervised_windows(series: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    for i in range(len(series) - seq_len):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Shape required by PyTorch recurrent layers: (batch, sequence, features)
    X = X[..., None]
    y = y[..., None]

    return X, y


class SequenceRegressor(nn.Module):
    def __init__(
        self,
        cell_type: str,
        input_size: int = 1,
        hidden_size: int = 48,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.cell_type = cell_type.upper()

        if self.cell_type == "RNN":
            self.recurrent = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity="tanh",
            )
        elif self.cell_type == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif self.cell_type == "GRU":
            self.recurrent = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError("cell_type must be one of: RNN, LSTM, GRU")

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    n = len(X)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, 1)
    x_scaler.fit(X_train_2d)

    X_train_scaled = x_scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val_scaled = x_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test_scaled = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    def loader(X_np, y_np, shuffle):
        dataset = TensorDataset(
            torch.tensor(X_np, dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    loaders = (
        loader(X_train_scaled, y_train_scaled, True),
        loader(X_val_scaled, y_val_scaled, False),
        loader(X_test_scaled, y_test_scaled, False),
    )

    scalers_and_data = {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "y_test_original": y_test,
    }

    return loaders[0], loaders[1], loaders[2], scalers_and_data


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            out = model(xb)
            loss = loss_fn(out, yb)

            total_loss += loss.item() * xb.size(0)
            preds.append(out.cpu().numpy())
            targets.append(yb.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)

    return total_loss / len(y_true), y_true, y_pred


def train_one_model(
    cell_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame]:
    model = SequenceRegressor(cell_type=cell_type).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    history = []
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        last_grad_norm = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()

            total_train_loss += loss.item() * xb.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)
        val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)

        history.append(
            {
                "model": cell_type,
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
                "last_batch_grad_norm_before_clipping": last_grad_norm,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{cell_type:4s} | epoch {epoch:03d} | "
                f"train_mse={train_loss:.5f} | val_mse={val_loss:.5f} | "
                f"grad_norm={last_grad_norm:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history)


def inverse_transform(y_scaler: StandardScaler, y: np.ndarray) -> np.ndarray:
    return y_scaler.inverse_transform(y.reshape(-1, 1)).ravel()


def plot_training_curves(history_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4.8))

    for model_name, group in history_df.groupby("model"):
        plt.plot(group["epoch"], group["val_mse"], marker="o", label=f"{model_name} validation MSE")

    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title("RNN vs LSTM vs GRU: Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = OUT_DIR / "rnn_lstm_gru_training_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_predictions(predictions_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))

    sample = predictions_df.head(250)
    plt.plot(sample["time_index"], sample["true_value"], label="True value")

    for col in ["RNN_prediction", "LSTM_prediction", "GRU_prediction"]:
        plt.plot(sample["time_index"], sample[col], label=col.replace("_", " "))

    plt.xlabel("Test time index")
    plt.ylabel("Signal value")
    plt.title("Time-Series Forecasting: True vs Predicted Values")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = OUT_DIR / "rnn_lstm_gru_test_predictions.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-points", type=int, default=2400)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    series = generate_time_series(n_points=args.n_points)
    X, y = make_supervised_windows(series, seq_len=args.seq_len)

    train_loader, val_loader, test_loader, aux = make_loaders(X, y, batch_size=args.batch_size)

    all_history = []
    metrics_rows = []
    prediction_table = None

    loss_fn = nn.MSELoss()

    for cell_type in ["RNN", "LSTM", "GRU"]:
        model, history_df = train_one_model(
            cell_type=cell_type,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )

        all_history.append(history_df)

        test_loss, y_true_scaled, y_pred_scaled = evaluate(model, test_loader, loss_fn, device)

        y_true = inverse_transform(aux["y_scaler"], y_true_scaled)
        y_pred = inverse_transform(aux["y_scaler"], y_pred_scaled)

        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics_rows.append(
            {
                "model": cell_type,
                "test_mse_scaled": test_loss,
                "test_rmse_original_scale": rmse,
                "test_mae_original_scale": mae,
                "test_r2_original_scale": r2,
            }
        )

        torch.save(model.state_dict(), OUT_DIR / f"best_{cell_type}.pt")

        if prediction_table is None:
            prediction_table = pd.DataFrame(
                {
                    "time_index": np.arange(len(y_true)),
                    "true_value": y_true,
                    f"{cell_type}_prediction": y_pred,
                }
            )
        else:
            prediction_table[f"{cell_type}_prediction"] = y_pred

    history_all_df = pd.concat(all_history, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)

    plot_training_curves(history_all_df)
    plot_predictions(prediction_table)

    output_excel = OUT_DIR / "rnn_lstm_gru_results.xlsx"
    with pd.ExcelWriter(output_excel) as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        history_all_df.to_excel(writer, sheet_name="training_history", index=False)
        prediction_table.to_excel(writer, sheet_name="test_predictions", index=False)
        pd.DataFrame({"series": series}).to_excel(writer, sheet_name="raw_series", index=False)

    print("\n=== RNN/LSTM/GRU Time-Series Results ===")
    print(metrics_df)
    print(f"\n[Saved] {output_excel}")


if __name__ == "__main__":
    main()
