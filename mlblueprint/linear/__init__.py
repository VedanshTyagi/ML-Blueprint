"""
Linear models family.

This family contains algorithms that learn linear relationships between
features and targets, including both regression and classification models.

Algorithms
----------
- LinearRegression: Linear regression with gradient descent
- LogisticRegression: Binary classification with the sigmoid and log-loss
"""

from .linear_regression import LinearRegression, LinearRegressionScratch
from .logistic_regression import LogisticRegression, LogisticRegressionScratch

__all__ = [
    "LinearRegression",
    "LinearRegressionScratch",
    "LogisticRegression",
    "LogisticRegressionScratch",
]
