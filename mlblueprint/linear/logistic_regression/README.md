# Logistic Regression

Family: linear
Completion & Scope: `complete`
Maintainers: @Aryan1092raj

Binary classification using a linear model passed through the sigmoid function, trained by gradient descent on log-loss. Teaches numerical stability of the loss, gradient computation, and how a simple linear model becomes a probability estimator.

## Files

| Version | File | Done? |
|---|---|---|
| Plain Python | `scratch.py` | yes |
| Fast version | `numpy_impl.py` | yes |
| PyTorch | `torch_impl.py` | no |

## Docs

[Intuition](docs/intuition.md), [Derivation](docs/derivation.md),
[Complexity](docs/complexity.md), [References](docs/references.md)

## Usage

```python
import numpy as np

from mlblueprint.linear import LogisticRegression

X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression(lr=0.5, n_iters=2000, lam=0.0, random_state=42).fit(X, y)

model.predict(np.array([[1.5]]))  # array([1])
model.predict_proba(np.array([[1.5]]))  # array([0.97...])
model.loss_history_[-1] < model.loss_history_[0]  # True
```

`fit` learns `w_` (weights) and `b_` (intercept), and records the regularized
loss (`log-loss + lam * ||w||^2`, see derivation section 8) at every iteration
in `loss_history_`. Labels must be exactly 0 and 1. Set `lam > 0` for L2
regularization; the bias is never regularized.

The pure-Python version takes and returns lists instead of arrays and accepts
the same hyperparameters:

```python
from mlblueprint.linear import LogisticRegressionScratch

model = LogisticRegressionScratch(lr=0.5, n_iters=2000, lam=0.0, random_state=42)
model.fit([[-2.0], [-1.0], [1.0], [2.0]], [0, 0, 1, 1])
```

Run `python -m mlblueprint.linear.logistic_regression.example` for a worked end-to-end run.
