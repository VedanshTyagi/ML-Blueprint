# Logistic Regression — Derivation

## 0. Notation

One symbol, one meaning throughout. Bold is a vector over all $n$ samples,
plain with $(i)$ is example $i$.

| Symbol | Means | Code name |
|---|---|---|
| $\mathcal{L}$ | likelihood (only ever the product) | — |
| $L$ | mean unregularized log-loss | — |
| $\ell_i$ | per-example contribution, $L = \frac{1}{n}\sum_i \ell_i$ | — |
| $J$ | regularized objective, $J = L + \lambda\lVert w\rVert^2$ (§8 only) | `loss_history_` |
| $z^{(i)}$, $\mathbf{z}$ | scalar logit / vector of all logits | `z_list`, `z` |
| $\hat{y}^{(i)}$, $\hat{\mathbf{y}}$ | scalar probability / vector of all probabilities | `y_hat` |
| $\alpha$ | learning rate | `lr` |
| $\lambda$ | L2 strength | `lam` |

## 1. The Model

For a dataset with $n$ samples and $d$ features, the logistic regression model predicts the probability of the positive class as:

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

where the linear predictor $z$ is:

$$z = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b$$

In vectorized notation:

$$z = Xw + b$$

$$\hat{y} = \sigma(Xw + b)$$

Where:
- **$X$** is the input feature matrix of shape $(n, d)$
- **$w$** is the weight vector of shape $(d,)$
- **$b$** is the bias term (scalar)
- **$n$** is the number of training examples
- **$d$** is the number of features
- **$\sigma(\cdot)$** is the sigmoid function
- **$\hat{y}$** is the predicted probability of class 1

## 2. The Loss Function — Bernoulli MLE

We assume each label $y^{(i)} \in \{0, 1\}$ is drawn from a Bernoulli distribution parameterized by $\hat{y}^{(i)}$:

$$P(y^{(i)} \mid x^{(i)}) = (\hat{y}^{(i)})^{y^{(i)}} (1 - \hat{y}^{(i)})^{1 - y^{(i)}}$$

The likelihood over all $n$ independent examples is:

$$\mathcal{L}(w, b) = \prod_{i=1}^{n} (\hat{y}^{(i)})^{y^{(i)}} (1 - \hat{y}^{(i)})^{1 - y^{(i)}}$$

Taking the log:

$$\log \mathcal{L}(w, b) = \sum_{i=1}^{n} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

We minimize the **negative average log-likelihood**, giving the binary cross-entropy loss:

$$L(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

## 3. Computing the Gradients

### 3.1 Derivative of the sigmoid

A useful identity:

$$\sigma'(z) = \frac{d}{dz} \frac{1}{1 + e^{-z}} = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)(1 - \sigma(z))$$

### 3.2 Derivative of loss with respect to $z$

For a single example $i$, the loss is:

$$\ell_i = - \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

Derivative with respect to $\hat{y}^{(i)}$:

$$\frac{\partial \ell_i}{\partial \hat{y}^{(i)}} = -\frac{y^{(i)}}{\hat{y}^{(i)}} + \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} = \frac{\hat{y}^{(i)} - y^{(i)}}{\hat{y}^{(i)}(1 - \hat{y}^{(i)})}$$

Chain rule through the sigmoid:

$$\frac{\partial \ell_i}{\partial z^{(i)}} = \frac{\partial \ell_i}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}}$$

$$= \frac{\hat{y}^{(i)} - y^{(i)}}{\hat{y}^{(i)}(1 - \hat{y}^{(i)})} \cdot \hat{y}^{(i)}(1 - \hat{y}^{(i)})$$

$$= \hat{y}^{(i)} - y^{(i)}$$

The sigmoid terms cancel exactly, leaving just the prediction error.

### 3.3 Derivative with respect to Weights ($w$)

$$\frac{\partial \ell_i}{\partial w_j} = \frac{\partial \ell_i}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w_j} = (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}$$

Averaging over all $n$ examples:

$$\frac{\partial L}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}$$

In vectorized form:

$$\frac{\partial L}{\partial w} = \frac{1}{n} X^T (\hat{y} - y)$$

### 3.4 Derivative with respect to Bias ($b$)

$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})$$

## 4. Convexity

The Hessian of the loss with respect to $w$ is:

$$\nabla_w^2 L = \frac{1}{n} X^T R X$$

where $R = \operatorname{diag}\left(\hat{y}^{(i)}(1 - \hat{y}^{(i)})\right)$ is a diagonal matrix with non-negative entries. For any vector $v$:

$$v^T (\nabla_w^2 L) v = \frac{1}{n} (X v)^T R (X v) \ge 0$$

Thus $L$ is convex in $w$.

## 5. Gradient Descent Update Rule

We update parameters by moving opposite to the gradient:

$$w := w - \alpha \frac{\partial L}{\partial w}$$

$$b := b - \alpha \frac{\partial L}{\partial b}$$

Substituting the gradients:

$$w := w - \frac{\alpha}{n} X^T (\hat{y} - y)$$

$$b := b - \frac{\alpha}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})$$

We repeat until convergence (loss stops decreasing significantly).

## 6. Prediction

For new inputs $X_{\text{new}}$, the predicted probability is:

$$\hat{y} = \sigma(X_{\text{new}} w + b)$$

The class prediction uses threshold 0.5:

$$\text{class} = \begin{cases} 1 & \text{if } \hat{y} \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

Since $\sigma(z) \ge 0.5 \iff z \ge 0$, the decision boundary is:

$$w^T x + b = 0$$

This confirms logistic regression is a **linear classifier** — the boundary in feature space is a hyperplane.

## 7. Numerical Stability Note

The gradient simplifies to a compact form, while the loss itself should be evaluated carefully for numerical stability. The naive expression $\log(\sigma(z))$ can overflow. In implementation, use the log-sum-exp trick or equivalently:

$$\ell(z, y) = \max(z, 0) - z y + \log(1 + e^{-|z|})$$

This avoids computing $\sigma(z)$ directly when $z$ is large in magnitude. It never exponentiates a positive number: both $\exp(-z)$ for $z \ge 0$ and $\exp(z)$ for $z < 0$ are at most 1. The same care applies to the sigmoid itself — branch on the sign of $z$ and only exponentiate the non-positive side, since evaluating both branches of a vectorized conditional first still overflows on the unused side.

## 8. L2 Regularization (the `lam` Option)

The implementation optionally minimizes the regularized objective:

$$J(w, b) = L(w, b) + \lambda \|w\|^2 = L(w, b) + \lambda \sum_{j=1}^{d} w_j^2$$

where $\lambda$ is the `lam` hyperparameter ($\lambda = 0$ recovers plain logistic regression). The bias $b$ is not regularized, so the boundary can shift freely while large weights are discouraged.

Since $\frac{\partial}{\partial w} \lambda \|w\|^2 = 2\lambda w$, the weight gradient gains one term and the bias gradient is unchanged:

$$\frac{\partial J}{\partial w} = \frac{1}{n} X^T (\hat{y} - y) + 2\lambda w$$

$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})$$

The update is therefore identical to section 5 with the extra $2\lambda w$ pull toward zero. Adding $\lambda I$-scaled curvature keeps the problem convex, and `loss_history_` records this regularized value $J$, not the raw $L$.
