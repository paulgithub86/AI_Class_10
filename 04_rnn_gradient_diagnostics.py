from __future__ import annotations

"""
Experiment 4: RNN gradient diagnostics.

This script uses a long-memory sequence classification task:
Given a sequence, predict whether the first value was positive.

This exposes the difficulty of preserving information across many time steps.
The experiment compares vanilla RNN, LSTM, and GRU, with and without gradient clipping.
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

from sklearn.metrics import accuracy_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path("outputs") / "04_rnn_gradients"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_long_memory_dataset(
    n_samples: int = 2500,
    seq_len: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)

    X = rng.normal(0.0, 1.0, size=(n_samples, seq_len, 1)).astype(np.float32)

    # Label depends only on the first time step.
    y = (X[:, 0, 0] > 0).astype(np.int64)

    # Add distractor structure to later positions.
    X[:, 1:, 0] += 0.15 * rng.normal(size=(n_samples, seq_len - 1)).astype(np.float32)

    return X, y


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        cell_type: str,
        hidden_size: int = 48,
    ) -> None:
        super().__init__()

        self.cell_type = cell_type.upper()

        if self.cell_type == "RNN":
            self.recurrent = nn.RNN(
                input_size=1,
                hidden_size=hidden_size,
                batch_first=True,
                nonlinearity="tanh",
            )
        elif self.cell_type == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                batch_first=True,
            )
        elif self.cell_type == "GRU":
            self.recurrent = nn.GRU(
                input_size=1,
                hidden_size=hidden_size,
                batch_first=True,
            )
        else:
            raise ValueError("cell_type must be one of: RNN, LSTM, GRU")

        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(x)
        last_hidden = output[:, -1, :]
        return self.classifier(last_hidden)


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    n = len(X)
    split = int(0.75 * n)

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            total_loss += loss.item() * xb.size(0)
            all_true.append(yb.cpu().numpy())
            all_pred.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    return total_loss / len(y_true), accuracy_score(y_true, y_pred)


def train_variant(
    variant_name: str,
    cell_type: str,
    use_gradient_clipping: bool,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> pd.DataFrame:
    model = SequenceClassifier(cell_type=cell_type).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rows = []

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        grad_norms_before_clip = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()

            grad_norm_before = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_before += p.grad.detach().norm().item() ** 2
            grad_norm_before = grad_norm_before ** 0.5
            grad_norms_before_clip.append(grad_norm_before)

            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        test_loss, test_acc = evaluate(model, test_loader, device)

        rows.append(
            {
                "variant": variant_name,
                "cell_type": cell_type,
                "uses_gradient_clipping": use_gradient_clipping,
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "mean_gradient_norm_before_clipping": float(np.mean(grad_norms_before_clip)),
                "max_gradient_norm_before_clipping": float(np.max(grad_norms_before_clip)),
            }
        )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"{variant_name:20s} | epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | test_acc={test_acc:.4f} | "
                f"mean_grad_norm={np.mean(grad_norms_before_clip):.4f}"
            )

    return pd.DataFrame(rows)


def plot_diagnostics(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8.5, 5))

    for variant, group in df.groupby("variant"):
        plt.plot(group["epoch"], group["test_accuracy"], marker="o", label=variant)

    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.ylim(0, 1.0)
    plt.title("Long-Memory Sequence Task: Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = OUT_DIR / "rnn_gradient_diagnostic_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")

    plt.figure(figsize=(8.5, 5))

    for variant, group in df.groupby("variant"):
        plt.plot(
            group["epoch"],
            group["mean_gradient_norm_before_clipping"],
            marker="o",
            label=variant,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Mean gradient norm before clipping")
    plt.title("Gradient Norms Before Optional Clipping")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = OUT_DIR / "rnn_gradient_norms.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=80)
    parser.add_argument("--n-samples", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    X, y = make_long_memory_dataset(n_samples=args.n_samples, seq_len=args.seq_len)
    train_loader, test_loader = make_loaders(X, y, batch_size=args.batch_size)

    variants = [
        {
            "variant_name": "RNN_no_clipping",
            "cell_type": "RNN",
            "use_gradient_clipping": False,
        },
        {
            "variant_name": "RNN_with_clipping",
            "cell_type": "RNN",
            "use_gradient_clipping": True,
        },
        {
            "variant_name": "LSTM_with_clipping",
            "cell_type": "LSTM",
            "use_gradient_clipping": True,
        },
        {
            "variant_name": "GRU_with_clipping",
            "cell_type": "GRU",
            "use_gradient_clipping": True,
        },
    ]

    all_results = []

    for cfg in variants:
        result_df = train_variant(
            variant_name=cfg["variant_name"],
            cell_type=cfg["cell_type"],
            use_gradient_clipping=cfg["use_gradient_clipping"],
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
        all_results.append(result_df)

    results_df = pd.concat(all_results, ignore_index=True)

    final_summary_df = (
        results_df.sort_values("epoch")
        .groupby("variant")
        .tail(1)
        .reset_index(drop=True)
    )

    plot_diagnostics(results_df)

    output_excel = OUT_DIR / "rnn_gradient_diagnostics_results.xlsx"
    with pd.ExcelWriter(output_excel) as writer:
        final_summary_df.to_excel(writer, sheet_name="final_summary", index=False)
        results_df.to_excel(writer, sheet_name="epoch_history", index=False)

    print("\n=== RNN Gradient Diagnostics Final Summary ===")
    print(final_summary_df)
    print(f"\n[Saved] {output_excel}")


if __name__ == "__main__":
    main()
