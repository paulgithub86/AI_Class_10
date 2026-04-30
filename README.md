# AI Class 10 Practical Session: CNNs and RNNs

## Course Context

**Course:** CS641 Artificial Intelligence  
**Lecture:** Class 10 — Deep Learning: Convolutional and Recurrent Neural Networks  
**Instructor:** Dr. A. Paul  
**Topic:** Convolutional Neural Networks, Recurrent Neural Networks, LSTM, GRU, BPTT, and gradient-flow problems.

This GitHub-ready practical session follows the Class 10 lecture topics. It begins with the mathematical operations behind CNNs, then trains a small CNN for image classification, and finally studies recurrent models for sequence/time-series learning.

The session is intentionally split into independent Python files so each experiment can be run, explained, modified, and graded separately.

---

## Learning Objectives

After completing this practical session, students should be able to:

1. Explain why CNNs are suitable for grid-like image data.
2. Implement convolution, padding, stride, and pooling operations from scratch.
3. Train a small CNN on CIFAR-10 or a clearly marked offline fallback dataset.
4. Compare CNNs with dense neural networks for image data.
5. Explain why RNNs are suitable for sequential data.
6. Train vanilla RNN, LSTM, and GRU models on a time-series forecasting task.
7. Interpret Backpropagation Through Time through gradient-flow diagnostics.
8. Identify vanishing/exploding gradients and apply practical remedies such as gradient clipping.

---

## Repository Structure

```text
AI_Class_10/
│
├── README.md
├── requirements.txt
├── 01_cnn_core_ops_from_scratch.py
├── 02_train_cnn_cifar10.py
├── 03_rnn_lstm_gru_timeseries.py
├── 04_rnn_gradient_diagnostics.py
└── run_all_experiments.py
```

---

## Environment Setup

Recommended Conda environment:

```bash
conda create -n ai_class10 python=3.11 -y
conda activate ai_class10
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If your machine has a CUDA-enabled GPU and you want a GPU-specific PyTorch build, install PyTorch from the official PyTorch installation selector first, then run:

```bash
pip install -r requirements.txt
```

---

## How to Run

Run each experiment independently:

```bash
python 01_cnn_core_ops_from_scratch.py
python 02_train_cnn_cifar10.py
python 03_rnn_lstm_gru_timeseries.py
python 04_rnn_gradient_diagnostics.py
```

Run all experiments sequentially:

```bash
python run_all_experiments.py
```

For faster classroom execution, the CNN and RNN scripts expose command-line options:

```bash
python 02_train_cnn_cifar10.py --epochs 2 --train-subset 5000 --test-subset 1000
python 03_rnn_lstm_gru_timeseries.py --epochs 25
python 04_rnn_gradient_diagnostics.py --epochs 12
```

All results are saved in:

```text
outputs/
```

---

# Experiment 1: CNN Core Operations from Scratch

**File:** `01_cnn_core_ops_from_scratch.py`

## Purpose

This experiment implements the core CNN operations without PyTorch:

1. 2D convolution
2. Zero padding
3. Stride
4. ReLU activation
5. Max pooling
6. Average pooling

## What Students Learn

Students see that convolution is not magic. A small kernel slides over an image and produces a feature map by element-wise multiplication and summation.

The script creates a synthetic image containing simple geometric structures and applies handcrafted kernels such as:

1. Vertical edge detector
2. Horizontal edge detector
3. Sharpening kernel
4. Blurring kernel

## Generated Outputs

```text
outputs/01_core_ops/cnn_core_ops_feature_maps.png
outputs/01_core_ops/cnn_core_ops_results.xlsx
```

---

# Experiment 2: CNN for Image Classification

**File:** `02_train_cnn_cifar10.py`

## Purpose

This experiment trains a small CNN on CIFAR-10 and compares it with a dense MLP baseline.

## Dataset

The script first tries to load CIFAR-10 through `torchvision.datasets.CIFAR10`.

If CIFAR-10 cannot be downloaded because the machine is offline, the script falls back to `torchvision.datasets.FakeData`. The fallback is clearly reported in the console and in the Excel result file. The fallback is only for verifying that the training pipeline works offline; its accuracy should not be interpreted scientifically.

## Models

### Tiny CNN

```text
Input: 3 x 32 x 32 image
Conv block 1: Conv2d -> BatchNorm -> ReLU -> MaxPool
Conv block 2: Conv2d -> BatchNorm -> ReLU -> MaxPool
Conv block 3: Conv2d -> BatchNorm -> ReLU -> AdaptiveAvgPool
Classifier: Linear layers
```

### Dense MLP Baseline

```text
Flatten image -> Dense layers -> Output classes
```

## What Students Learn

1. Local connectivity
2. Weight sharing
3. Translation-aware feature extraction
4. Convolution/pooling hierarchy
5. Why CNNs are usually better suited than dense MLPs for images

## Generated Outputs

```text
outputs/02_cifar10_cnn/cnn_vs_mlp_training_curves.png
outputs/02_cifar10_cnn/cnn_confusion_matrix.png
outputs/02_cifar10_cnn/cnn_sample_predictions.png
outputs/02_cifar10_cnn/cnn_first_layer_feature_maps.png
outputs/02_cifar10_cnn/cifar10_cnn_results.xlsx
outputs/02_cifar10_cnn/tiny_cnn_model.pt
```

---

# Experiment 3: RNN, LSTM, and GRU for Time-Series Forecasting

**File:** `03_rnn_lstm_gru_timeseries.py`

## Purpose

This experiment compares vanilla RNN, LSTM, and GRU models on a synthetic time-series forecasting task.

The generated signal contains:

1. Periodic components
2. Trend
3. Noise
4. Nonlinear structure

The task is to predict the next value from a fixed-length input sequence.

## Models

1. Vanilla RNN
2. LSTM
3. GRU

## What Students Learn

1. Hidden-state recurrence
2. Parameter sharing over time
3. Sequence-to-one prediction
4. Why gated models often train more reliably than vanilla RNNs
5. Gradient clipping for recurrent models

## Generated Outputs

```text
outputs/03_timeseries_rnn/rnn_lstm_gru_training_curves.png
outputs/03_timeseries_rnn/rnn_lstm_gru_test_predictions.png
outputs/03_timeseries_rnn/rnn_lstm_gru_results.xlsx
outputs/03_timeseries_rnn/best_RNN.pt
outputs/03_timeseries_rnn/best_LSTM.pt
outputs/03_timeseries_rnn/best_GRU.pt
```

---

# Experiment 4: RNN Gradient Diagnostics

**File:** `04_rnn_gradient_diagnostics.py`

## Purpose

This experiment demonstrates vanishing and exploding gradients in recurrent networks.

It uses a synthetic long-memory classification task:

```text
Given a sequence of random numbers, predict whether the first value in the sequence was positive.
```

This is deliberately hard for vanilla RNNs when the sequence is long because the model must preserve early information until the end.

## Compared Models

1. Vanilla RNN without gradient clipping
2. Vanilla RNN with gradient clipping
3. LSTM with gradient clipping
4. GRU with gradient clipping

## What Students Learn

1. BPTT unfolds recurrence over time.
2. Gradients may shrink or grow across long sequences.
3. Gradient clipping stabilizes exploding gradients.
4. LSTM/GRU gates help preserve useful temporal information.

## Generated Outputs

```text
outputs/04_rnn_gradients/rnn_gradient_diagnostic_curves.png
outputs/04_rnn_gradients/rnn_gradient_norms.png
outputs/04_rnn_gradients/rnn_gradient_diagnostics_results.xlsx
```

---

## Suggested 3-Hour Lab Plan

| Time | Activity |
|---|---|
| 15 min | Review CNN and RNN motivation |
| 25 min | Run and explain Experiment 1 |
| 45 min | Run CNN/MLP image classification experiment |
| 20 min | Discuss CNN feature maps and confusion matrix |
| 45 min | Run RNN/LSTM/GRU time-series experiment |
| 30 min | Run gradient diagnostic experiment |
| 20 min | Discussion and student Q&A |

---

## Student Submission Requirements

Students should submit:

1. All generated figures.
2. All generated Excel result files.
3. A short explanation of convolution and pooling.
4. A comparison between the CNN and dense MLP baseline.
5. A short explanation of why LSTM/GRU can handle long-term dependencies better than vanilla RNN.
6. A discussion of the gradient-norm plots.
7. Modified results after changing at least one hyperparameter.

---

## Suggested Student Exercises

### Exercise 1

In `01_cnn_core_ops_from_scratch.py`, add a diagonal edge-detection kernel.

### Exercise 2

In `02_train_cnn_cifar10.py`, change the number of filters from:

```python
32, 64, 128
```

to:

```python
16, 32, 64
```

Compare parameter count and accuracy.

### Exercise 3

In `02_train_cnn_cifar10.py`, remove data augmentation and compare validation/test accuracy.

### Exercise 4

In `03_rnn_lstm_gru_timeseries.py`, change the sequence length from 30 to 60.

### Exercise 5

In `04_rnn_gradient_diagnostics.py`, disable gradient clipping and observe whether the gradient norm becomes unstable.

---

## GitHub Recommendation

Suggested repository/folder name:

```text
AI_Class_10
```

Suggested first commit:

```bash
git init
git add README.md requirements.txt *.py
git commit -m "Add AI Class 10 CNN and RNN practical session"
```

---

## Academic Integrity Note

Students should not only execute the scripts. They should be able to explain the model architecture, the learning objective, the loss curves, and the reason for gradient instability in deep or recurrent neural networks.
