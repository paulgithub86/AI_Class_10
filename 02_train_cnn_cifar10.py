from __future__ import annotations

"""
Experiment 2: Train a small CNN on CIFAR-10 and compare it with a dense MLP.

The script first tries to use CIFAR-10. If CIFAR-10 cannot be downloaded because
the machine is offline, it falls back to torchvision FakeData and clearly reports
that the dataset is synthetic. FakeData is useful only for code verification.
"""

import argparse
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

try:
    import torchvision
    from torchvision import datasets, transforms
except ImportError as exc:
    raise ImportError(
        "torchvision is required for this script. Install it using: pip install torchvision"
    ) from exc

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


OUT_DIR = Path("outputs") / "02_cifar10_cnn"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyCIFARCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(64, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DenseImageMLP(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def subset_dataset(dataset, max_items: int | None):
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset

    indices = list(range(len(dataset)))
    rng = random.Random(SEED)
    rng.shuffle(indices)
    return Subset(dataset, indices[:max_items])


def load_cifar10_or_fallback(data_dir: Path, train_subset: int, test_subset: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    dataset_name = "CIFAR-10"
    class_names = CIFAR10_CLASSES

    try:
        train_full = datasets.CIFAR10(
            root=str(data_dir), train=True, download=True, transform=train_transform
        )
        test_full = datasets.CIFAR10(
            root=str(data_dir), train=False, download=True, transform=test_transform
        )
    except Exception as exc:
        print("\n[Warning] CIFAR-10 could not be loaded.")
        print(f"[Warning] Reason: {exc}")
        print("[Warning] Falling back to torchvision.datasets.FakeData.")
        print("[Warning] Fallback accuracy is not scientifically meaningful.\n")

        dataset_name = "FakeData_offline_fallback"
        class_names = [f"class_{i}" for i in range(10)]

        train_full = datasets.FakeData(
            size=max(train_subset, 1000),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=test_transform,
            random_offset=0,
        )
        test_full = datasets.FakeData(
            size=max(test_subset, 500),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=test_transform,
            random_offset=10_000,
        )

    train_ds = subset_dataset(train_full, train_subset)
    test_ds = subset_dataset(test_full, test_subset)

    val_size = max(1, int(0.15 * len(train_ds)))
    train_size = len(train_ds) - val_size

    train_ds, val_ds = random_split(
        train_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    return dataset_name, class_names, train_ds, val_ds, test_ds


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    all_true = []
    all_pred = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        all_true.append(yb.detach().cpu().numpy())
        all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    return total_loss / len(y_true), accuracy_score(y_true, y_pred)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()

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
            all_true.append(yb.detach().cpu().numpy())
            all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    return total_loss / len(y_true), accuracy_score(y_true, y_pred), y_true, y_pred


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame]:
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)

        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "num_parameters": count_parameters(model),
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"{model_name:8s} | epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history)


def plot_training_curves(history_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4.8))

    for model_name, group in history_df.groupby("model"):
        plt.plot(group["epoch"], group["val_accuracy"], marker="o", label=f"{model_name} val acc")

    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.ylim(0, 1.0)
    plt.title("CNN vs Dense MLP on Image Classification")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = OUT_DIR / "cnn_vs_mlp_training_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], accuracy: float) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(f"Tiny CNN Confusion Matrix | test accuracy={accuracy:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    tick_positions = np.arange(len(class_names))
    plt.xticks(tick_positions, class_names, rotation=45, ha="right")
    plt.yticks(tick_positions, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    path = OUT_DIR / "cnn_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def denormalize_cifar_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return torch.clamp(x.cpu() * std + mean, 0, 1)


def plot_sample_predictions(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    max_images: int = 16,
) -> None:
    model.eval()
    xb, yb = next(iter(loader))
    xb_device = xb.to(device)

    with torch.no_grad():
        preds = model(xb_device).argmax(dim=1).cpu()

    n = min(max_images, xb.size(0))
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(10, 2.8 * rows))

    for idx in range(n):
        img = denormalize_cifar_tensor(xb[idx]).permute(1, 2, 0).numpy()
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.title(f"T: {class_names[int(yb[idx])]}\nP: {class_names[int(preds[idx])]}", fontsize=8)
        plt.axis("off")

    path = OUT_DIR / "cnn_sample_predictions.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def plot_first_layer_feature_maps(
    model: TinyCIFARCNN,
    loader: DataLoader,
    device: torch.device,
    max_maps: int = 12,
) -> None:
    model.eval()

    xb, _ = next(iter(loader))
    x_one = xb[:1].to(device)

    first_conv = None
    for module in model.features:
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break

    if first_conv is None:
        return

    with torch.no_grad():
        fmap = first_conv(x_one).detach().cpu()[0]

    n = min(max_maps, fmap.shape[0])
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(10, 2.5 * rows))

    for idx in range(n):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(fmap[idx].numpy(), cmap="gray")
        plt.title(f"Feature map {idx}")
        plt.axis("off")

    path = OUT_DIR / "cnn_first_layer_feature_maps.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-subset", type=int, default=5000)
    parser.add_argument("--test-subset", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    dataset_name, class_names, train_ds, val_ds, test_ds = load_cifar10_or_fallback(
        data_dir=Path(args.data_dir),
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )

    print(f"Dataset used: {dataset_name}")
    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cnn = TinyCIFARCNN(num_classes=10)
    mlp = DenseImageMLP(num_classes=10)

    print(f"Tiny CNN parameters: {count_parameters(cnn):,}")
    print(f"Dense MLP parameters: {count_parameters(mlp):,}")

    cnn, cnn_history = train_model("TinyCNN", cnn, train_loader, val_loader, args.epochs, args.lr, device)
    mlp, mlp_history = train_model("DenseMLP", mlp, train_loader, val_loader, args.epochs, args.lr, device)

    history_df = pd.concat([cnn_history, mlp_history], ignore_index=True)

    loss_fn = nn.CrossEntropyLoss()
    cnn_test_loss, cnn_test_acc, y_true, y_pred = evaluate(cnn, test_loader, loss_fn, device)
    mlp_test_loss, mlp_test_acc, _, _ = evaluate(mlp.to(device), test_loader, loss_fn, device)

    cm = confusion_matrix(y_true, y_pred)
    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()

    summary_df = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "model": "TinyCNN",
                "test_loss": cnn_test_loss,
                "test_accuracy": cnn_test_acc,
                "num_parameters": count_parameters(cnn),
            },
            {
                "dataset": dataset_name,
                "model": "DenseMLP",
                "test_loss": mlp_test_loss,
                "test_accuracy": mlp_test_acc,
                "num_parameters": count_parameters(mlp),
            },
        ]
    )

    plot_training_curves(history_df)
    plot_confusion_matrix(cm, class_names, cnn_test_acc)
    plot_sample_predictions(cnn, test_loader, class_names, device)
    plot_first_layer_feature_maps(cnn, test_loader, device)

    model_path = OUT_DIR / "tiny_cnn_model.pt"
    torch.save(cnn.state_dict(), model_path)

    output_excel = OUT_DIR / "cifar10_cnn_results.xlsx"
    with pd.ExcelWriter(output_excel) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        history_df.to_excel(writer, sheet_name="training_history", index=False)
        pd.DataFrame(cm).to_excel(writer, sheet_name="cnn_confusion_matrix", index=False)
        report_df.to_excel(writer, sheet_name="cnn_classification_report")

    print("\n=== CNN Image Classification Summary ===")
    print(summary_df)
    print(f"\n[Saved] {output_excel}")
    print(f"[Saved] {model_path}")


if __name__ == "__main__":
    main()
