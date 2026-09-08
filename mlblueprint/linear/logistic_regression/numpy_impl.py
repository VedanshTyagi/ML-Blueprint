"""NumPy implementation of Logistic Regression with L2 regularization."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mlblueprint.core.base import Predictor
from mlblueprint.core.random import check_random_state
from mlblueprint.core.validation import check_array, check_X_y


class LogisticRegression(Predictor):
    """Logistic Regression with L2 regularization, trained by gradient descent.

    Minimizes the regularized log-loss ``L(w, b) + lam * ||w||^2`` where
    ``L`` is the mean binary cross-entropy from ``docs/derivation.md``.
    The bias ``b`` is not regularized. See ``docs/derivation.md`` section 8
    for the regularized gradient.

    Parameters
    ----------
    lr : float
        Learning rate. Default 0.1.
    n_iters : int
        Number of gradient descent iterations. Default 1000.
    lam : float
        L2 regularization strength on ``w_``. Set to 0 for plain
        logistic regression. Default 0.0.
    random_state : int or numpy.random.Generator or None
        Seed for reproducibility.

    Attributes
    ----------
    w_ : ndarray of shape (n_features,)
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
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """Store hyperparameters; all real work happens in ``fit``."""
        self.lr = lr
        self.n_iters = n_iters
        self.lam = lam
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> LogisticRegression:
        """Train the model using gradient descent.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training input data.
        y : array-like, shape (n_samples,)
            Target values, exactly the labels 0 and 1.

        Returns
        -------
        self : LogisticRegression
            The fitted model.

        Raises
        ------
        ValueError
            If ``y`` does not contain exactly the labels 0 and 1.
        """
        X, y = check_X_y(X, y, y_numeric=False)

        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError(
                f"y must contain exactly the labels 0 and 1, but it contains "
                f"{classes.tolist()}. Map your labels to 0 and 1 before fitting, "
                "or use a multiclass model."
            )
        y = y.astype(np.float64)

        rng = check_random_state(self.random_state)
        n_samples, n_features = X.shape

        # init parameters
        self.w_ = rng.normal(scale=0.01, size=n_features)
        self.b_ = 0.0
        self.loss_history_ = []

        # training loop
        for _ in range(self.n_iters):
            # forward pass
            z = X @ self.w_ + self.b_
            y_hat = self._sigmoid(z)

            # Log-loss in the rearranged form from docs/derivation.md section 7.
            # Written this way it never exponentiates a positive number, so it
            # stays exact for large |z| instead of saturating to log(0) the
            # way clipping y_hat would.
            loss = np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))
            self.loss_history_.append(float(loss + self.lam * np.sum(self.w_**2)))

            # gradients of the regularized loss (derivation section 8)
            error = y_hat - y
            dw = (1 / n_samples) * (X.T @ error) + 2 * self.lam * self.w_
            db = (1 / n_samples) * np.sum(error)

            # update
            self.w_ -= self.lr * dw
            self.b_ -= self.lr * db

        return self

    def _sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Squash logits into probabilities without overflowing.

        Only ever exponentiates a non-positive number: ``exp(-z)`` for
        ``z >= 0`` and ``exp(z)`` for ``z < 0`` are both at most 1.
        Selecting the branch afterwards with ``np.where`` is not enough,
        both sides of ``np.where`` are evaluated first and the unused one
        still overflows.
        """
        out = np.empty_like(z, dtype=np.float64)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        out[~positive] = exp_z / (1.0 + exp_z)
        return out

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict probabilities of class 1.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to predict on.

        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted probabilities of class 1.
        """
        self._check_is_fitted()
        X = check_array(X)
        z = X @ self.w_ + self.b_
        return self._sigmoid(z)

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> NDArray[np.int_]:
        """Predict class labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to predict on.
        threshold : float, default=0.5
            Decision threshold for class 1.

        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
