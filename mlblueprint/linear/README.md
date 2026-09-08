# Linear Models

`mlblueprint.linear`

Algorithms that learn linear relationships between features and targets.

## Algorithms

| Algorithm | Status | Description |
|-----------|--------|-------------|
| [Linear Regression](linear_regression/) | `complete` | Predicts continuous values using a linear equation |
| [Logistic Regression](logistic_regression/) | `complete` | Predicts the probability of a binary class using the sigmoid |

## Common Properties

All linear models in this family:
- Learn weights (coefficients) for each feature
- Include a bias term (intercept)
- Are interpretable and fast to train
- Work well as baseline models before trying more complex algorithms