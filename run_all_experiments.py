from __future__ import annotations

"""
Run all AI Class 10 practical experiments sequentially.

The CNN and recurrent experiments are configured with moderate defaults.
You can run each script independently for more control.
"""

import subprocess
import sys
from pathlib import Path


COMMANDS = [
    [sys.executable, "01_cnn_core_ops_from_scratch.py"],
    [sys.executable, "02_train_cnn_cifar10.py", "--epochs", "2", "--train-subset", "3000", "--test-subset", "800"],
    [sys.executable, "03_rnn_lstm_gru_timeseries.py", "--epochs", "20"],
    [sys.executable, "04_rnn_gradient_diagnostics.py", "--epochs", "10"],
]


def main() -> None:
    root = Path(__file__).resolve().parent

    for command in COMMANDS:
        print("\n" + "=" * 90)
        print("Running:", " ".join(command))
        print("=" * 90)

        result = subprocess.run(command, cwd=root, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(command)}")

    print("\nAll AI Class 10 experiments completed successfully.")


if __name__ == "__main__":
    main()
