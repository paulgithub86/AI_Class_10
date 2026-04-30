from __future__ import annotations

"""
Experiment 1: CNN core operations from scratch.

This script implements convolution, ReLU, max pooling, and average pooling using NumPy.
It helps students understand the operations that later appear inside a CNN layer.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = Path("outputs") / "01_core_ops"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pad2d(x: np.ndarray, padding: int) -> np.ndarray:
    if padding == 0:
        return x.copy()
    return np.pad(x, pad_width=((padding, padding), (padding, padding)), mode="constant")


def conv2d_single_channel(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """
    Perform valid/same-style 2D convolution for one image channel.

    Technically, many deep-learning libraries implement cross-correlation rather than
    mathematically flipped convolution. For teaching CNNs, the sliding dot-product
    operation is the important concept.
    """
    x = pad2d(image, padding)
    kh, kw = kernel.shape
    h, w = x.shape

    out_h = ((h - kh) // stride) + 1
    out_w = ((w - kw) // stride) + 1

    output = np.zeros((out_h, out_w), dtype=float)

    for i in range(out_h):
        for j in range(out_w):
            patch = x[i * stride : i * stride + kh, j * stride : j * stride + kw]
            output[i, j] = np.sum(patch * kernel)

    return output


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def pool2d(
    feature_map: np.ndarray,
    pool_size: int = 2,
    stride: int = 2,
    mode: str = "max",
) -> np.ndarray:
    h, w = feature_map.shape
    out_h = ((h - pool_size) // stride) + 1
    out_w = ((w - pool_size) // stride) + 1

    output = np.zeros((out_h, out_w), dtype=float)

    for i in range(out_h):
        for j in range(out_w):
            patch = feature_map[
                i * stride : i * stride + pool_size,
                j * stride : j * stride + pool_size,
            ]

            if mode == "max":
                output[i, j] = np.max(patch)
            elif mode == "avg":
                output[i, j] = np.mean(patch)
            else:
                raise ValueError("mode must be either 'max' or 'avg'")

    return output


def create_synthetic_image(size: int = 32) -> np.ndarray:
    image = np.zeros((size, size), dtype=float)

    # Bright square
    image[6:18, 6:18] = 1.0

    # Medium-intensity rectangle
    image[12:28, 20:26] = 0.7

    # Diagonal line
    for idx in range(5, 27):
        image[idx, idx] = 1.0

    # Smooth gradient background
    gradient = np.linspace(0, 0.25, size)
    image += gradient.reshape(1, -1)

    return np.clip(image, 0, 1)


def plot_results(image: np.ndarray, results: dict[str, np.ndarray]) -> None:
    names = ["Input"] + list(results.keys())
    arrays = [image] + list(results.values())

    n = len(arrays)
    cols = 3
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(12, 4 * rows))

    for idx, (name, arr) in enumerate(zip(names, arrays), start=1):
        plt.subplot(rows, cols, idx)
        plt.imshow(arr, cmap="gray")
        plt.title(name)
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

    path = OUT_DIR / "cnn_core_ops_feature_maps.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def main() -> None:
    image = create_synthetic_image(size=32)

    kernels = {
        "vertical_edge": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
        "horizontal_edge": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
        "blur": np.ones((3, 3), dtype=float) / 9.0,
    }

    results = {}
    summary_rows = []

    for name, kernel in kernels.items():
        conv = conv2d_single_channel(image, kernel, stride=1, padding=1)
        activated = relu(conv)
        max_pooled = pool2d(activated, pool_size=2, stride=2, mode="max")
        avg_pooled = pool2d(activated, pool_size=2, stride=2, mode="avg")

        results[f"{name}_conv"] = conv
        results[f"{name}_relu"] = activated
        results[f"{name}_maxpool"] = max_pooled
        results[f"{name}_avgpool"] = avg_pooled

        summary_rows.append(
            {
                "kernel": name,
                "conv_shape": str(conv.shape),
                "conv_min": float(np.min(conv)),
                "conv_max": float(np.max(conv)),
                "relu_nonzero_count": int(np.count_nonzero(activated)),
                "maxpool_shape": str(max_pooled.shape),
                "avgpool_shape": str(avg_pooled.shape),
            }
        )

    # Plot a selected subset to keep the figure readable.
    selected = {
        "Vertical edge convolution": results["vertical_edge_conv"],
        "Vertical edge + ReLU": results["vertical_edge_relu"],
        "Vertical edge max pooling": results["vertical_edge_maxpool"],
        "Horizontal edge convolution": results["horizontal_edge_conv"],
        "Sharpen convolution": results["sharpen_conv"],
        "Blur convolution": results["blur_conv"],
    }

    plot_results(image, selected)

    output_excel = OUT_DIR / "cnn_core_ops_results.xlsx"
    with pd.ExcelWriter(output_excel) as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(image).to_excel(writer, sheet_name="input_image", index=False)

        for name, arr in selected.items():
            clean_name = name.lower().replace(" ", "_").replace("+", "plus")[:31]
            pd.DataFrame(arr).to_excel(writer, sheet_name=clean_name, index=False)

    print("\n=== CNN Core Operations Summary ===")
    print(pd.DataFrame(summary_rows))
    print(f"\n[Saved] {output_excel}")


if __name__ == "__main__":
    main()
