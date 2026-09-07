"""Pure Python implementation of Logistic Regression without external libraries.

Stdlib only, so it stays readable without NumPy. For speed use
:mod:`mlblueprint.linear.logistic_regression.numpy_impl` instead.
"""

from __future__ import annotations

import math
import random

from mlblueprint.core.base import Predictor


def sigmoid(z: float) -> float:
    """Squash a logit into a probability in (0, 1).

    Parameters
    ----------
    z : float
        Logit, ``w . x + b``.

    Returns
    -------
    float
        ``1 / (1 + exp(-z))``, evaluated without overflowing for large
        ``|z|`` by only ever exponentiating a non-positive number.
    """
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def log_loss(y_true: list[int | float], y_pred: list[float]) -> float:
    """Mean binary cross-entropy, with clipping for stability.

    Parameters
    ----------
    y_true : list of float
        True labels (0 or 1).
    y_pred : list of float
        Predicted probabilities of class 1.

    Returns
    -------
    float
        ``-(1/n) * sum(y*log(p) + (1-y)*log(1-p))``.
    """
    eps = 1e-15
    clipped = [max(eps, min(1 - eps, p)) for p in y_pred]
    return -sum(
        y * math.log(p) + (1 - y) * math.log(1 - p)
        for y, p in zip(y_true, clipped, strict=True)
    ) / len(y_true)


def _stable_loss_from_z(z_list: list[float], y_true: list[int | float]) -> float:
    """Mean log-loss directly from logits, without forming ``sigmoid(z)``.

    Uses ``max(z,0) - z*y + log1p(exp(-|z|))`` from ``docs/derivation.md``
    section 7, so large ``|z|`` stays exact instead of saturating to
    ``log(0)`` the way clipping ``y_hat`` would.
    """
    total = 0.0
    for z, y in zip(z_list, y_true, strict=True):
        total += max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))
    return total / len(z_list)


class LogisticRegression(Predictor):
    """Logistic Regression using plain Python lists and gradient descent.

    Mirrors the NumPy implementation with the same hyperparameters and update
    rule, so the two agree on the same data (up to random initialisation).

    Parameters
    ----------
    lr : float
        Learning rate. Default 0.1.
    n_iters : int
        Number of gradient descent iterations. Default 1000.
    lam : float
        L2 regularization strength on ``w_`` (bias is not regularized).
        Set to 0 for plain logistic regression. Default 0.0.
    random_state : int or None
        Seed for reproducibility. ``None`` seeds from the OS.

    Attributes
    ----------
    w_ : list of float
        Weights learned during fit.
    b_ : float
        Intercept learned during fit.
    loss_history_ : list of float
        Regularized loss at each iteration, oldest first.
    """

    def __init__(
        self,
        lr: float = 0.1,
        n_iters: int = 1000,
        lam: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        """Store hyperparameters; all real work happens in ``fit``."""
        self.lr = lr
        self.n_iters = n_iters
        self.lam = lam
        self.random_state = random_state

    def fit(self, X: list[list[float]], y: list[int | float]) -> LogisticRegression:
        """Train the model using gradient descent.

        Parameters
        ----------
        X : list of list of float, shape (n_samples, n_features)
            Training input data.
        y : list of float, shape (n_samples,)
            Target values, exactly the labels 0 and 1.

        Returns
        -------
        self : LogisticRegression
            The fitted model.
        """
        if len(X) == 0:
            raise ValueError("X is empty. Provide at least one sample to fit on.")
        if len(X) != len(y):
            raise ValueError(
                f"X has {len(X)} rows but y has {len(y)}. They need to match."
            )
        n_features = len(X[0])
        for row in X:
            if len(row) != n_features:
                raise ValueError(
                    "Every row of X must have the same number of features. "
                    f"Expected {n_features}, got {len(row)}. Reshape your input."
                )
        classes = set(y)
        if classes != {0, 1}:
            raise ValueError(
                f"y must contain exactly the labels 0 and 1, but it contains "
                f"{sorted(classes)}. Map your labels to 0 and 1 before fitting, "
                "or use a multiclass model."
            )

        n_samples = len(X)
        # ponytail: stdlib-only by design, so an isolated random.Random instead of
        # core.random.check_random_state (which needs NumPy). Same guarantee:
        # same seed -> same init, and never the shared global state.
        rng = random.Random(self.random_state)
        w = [rng.random() * 0.01 for _ in range(n_features)]
        b = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iters):
            z_list = [
                sum(X[i][j] * w[j] for j in range(n_features)) + b
                for i in range(n_samples)
            ]
            y_hat = [sigmoid(z) for z in z_list]
            self.loss_history_.append(
                _stable_loss_from_z(z_list, y) + self.lam * sum(v * v for v in w)
            )

            dw = [0.0 for _ in range(n_features)]
            db = 0.0
            for i in range(n_samples):
                error = y_hat[i] - y[i]
                for j in range(n_features):
                    dw[j] += X[i][j] * error
                db += error

            for j in range(n_features):
                w[j] -= self.lr * (dw[j] / n_samples + 2 * self.lam * w[j])
            b -= self.lr * db / n_samples

        self.w_ = w
        self.b_ = b
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        """Predict probabilities of class 1.

        Parameters
        ----------
        X : list of list of float, shape (n_samples, n_features)
            Input data to predict on.

        Returns
        -------
        list of float
            Predicted probabilities of class 1.
        """
        self._check_is_fitted()
        return [
            sigmoid(sum(row[j] * self.w_[j] for j in range(len(self.w_))) + self.b_)
            for row in X
        ]

    def predict(self, X: list[list[float]], threshold: float = 0.5) -> list[int]:
        """Predict class labels.

        Parameters
        ----------
        X : list of list of float, shape (n_samples, n_features)
            Input data to predict on.
        threshold : float, default=0.5
            Decision threshold for class 1.

        Returns
        -------
        list of int
            Predicted class labels (0 or 1).
        """
        probs = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probs]
