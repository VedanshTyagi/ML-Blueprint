# Logistic Regression — Complexity Analysis

## Overview

This document analyzes the time and space complexity of the Logistic Regression
implementation using Gradient Descent, for both `numpy_impl.py` (vectorized)
and `scratch.py` (plain-Python loops). Both do the same maths from
[derivation](derivation.md); they differ only by a constant factor.

## Notation

- **n** = number of training samples
- **d** = number of features
- **T** = number of training iterations (epochs)
- **m** = number of new samples passed to `predict`

## Time Complexity

### Training (fit)

Each iteration does three things in order. The forward pass forms `z = X @ w + b`: every one of the `n` samples dots `d` weights, so `O(n × d)`. The sigmoid and the stable log-loss touch each of the `n` predictions once, so `O(n)`. The gradient forms `X.T @ (y_hat - y)`: each of the `d` weights accumulates over `n` samples, so `O(n × d)` again, plus `O(d)` for the `2*lam*w` L2 term and the weight update.

| Operation | Complexity | Explanation |
|-----------|-----------|-------------|
| Forward pass (z = X @ w + b) | O(n × d) | n samples, each dots d weights |
| Sigmoid + stable loss | O(n) | One pass over the n logits |
| Compute gradients | O(n × d) | X.T @ error, plus O(d) L2 term |
| Update weights | O(d) | Subtract gradient from weights |
| **Per iteration** | **O(n × d)** | Matrix terms dominate the O(n) + O(d) extras |
| **Total training** | **O(T × n × d)** | T iterations of the above |

`scratch.py` is also `O(T × n × d)` — the same triple loop written out — but with a large Python-loop constant versus the NumPy matrix multiply, which is why `numpy_impl.py` is the fast version.

### Prediction (predict)

Prediction forms `X @ w + b` on `m` new samples and squashes each logit once.

| Operation | Complexity | Explanation |
|-----------|-----------|-------------|
| Single prediction | O(d) | Dot product + sigmoid |
| Batch prediction | O(m × d) | m new samples, each requires O(d) |

## Space Complexity

| Component | Space | Explanation |
|-----------|-------|-------------|
| Weights | O(d) | One weight per feature |
| Bias | O(1) | Single scalar |
| Input data | O(n × d) | Training matrix (not copied, just referenced) |
| Predictions (y_hat) | O(n) | Probabilities held during one iteration |
| Gradients | O(d) | One gradient per weight |
| Loss history | O(T) | One float per iteration in `loss_history_` |
| **Total** | **O(n × d + T)** | Input dominates unless T exceeds n × d |

## Summary

| Aspect | Complexity |
|--------|-----------|
| Training | O(T × n × d) |
| Prediction | O(m × d) |
| Space | O(n × d + T) |

## Notes

The complexity is identical to Linear Regression with Gradient Descent. The sigmoid
and log-loss add only O(n) operations per iteration, which is absorbed by the
O(n × d) matrix multiply term. The L2 term adds O(d) per iteration, likewise
absorbed. Memory differs from the naive account by the O(T) loss history, which
is the only thing that grows with iterations rather than data.
