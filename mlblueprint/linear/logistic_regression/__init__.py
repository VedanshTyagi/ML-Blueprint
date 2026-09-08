"""Logistic Regression.

Binary classification with a linear model passed through the sigmoid, trained by
gradient descent on the log-loss.
"""

from .numpy_impl import LogisticRegression
from .scratch import LogisticRegression as LogisticRegressionScratch

__all__ = [
    "LogisticRegression",
    "LogisticRegressionScratch",
]
