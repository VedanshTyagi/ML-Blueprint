"""Tests for the Logistic Regression implementations."""

import warnings

import numpy as np
import pytest

from mlblueprint.core.base import NotFittedError
from mlblueprint.core.datasets import make_blobs
from mlblueprint.core.metrics import accuracy_score
from mlblueprint.linear import LogisticRegression, LogisticRegressionScratch


def standardize(X):
    """Centre and scale features, which is what makes a fixed learning rate safe."""
    return (X - X.mean(axis=0)) / X.std(axis=0)


def log_loss(X, y, w, b):
    """Mean log-loss, in the same stable form numpy_impl uses."""
    z = X @ w + b
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


class TestLogisticRegression:
    """Correctness of the NumPy implementation."""

    def test_learns_a_separable_boundary(self):
        """0s sit left of the origin and 1s right, so the weight must be positive."""
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])

        model = LogisticRegression(lr=0.5, n_iters=2000, random_state=42).fit(X, y)

        assert model.w_[0] > 0
        np.testing.assert_array_equal(model.predict(X), y)

    def test_probabilities_agree_with_labels(self):
        """Predict is exactly predict_proba thresholded at 0.5, and stays in (0, 1)."""
        X, y = make_blobs(n_samples=100, centers=2, random_state=42)
        X = standardize(X)

        model = LogisticRegression(lr=0.1, n_iters=1000, random_state=42).fit(X, y)
        probs = model.predict_proba(X)

        assert np.all((probs > 0.0) & (probs < 1.0))
        np.testing.assert_array_equal(model.predict(X), (probs >= 0.5).astype(int))

    def test_loss_decreases_every_iteration(self):
        """Log-loss is convex, so a small enough step can only ever go downhill.

        This is the cheapest check that the gradient has the right sign and scale: a
        flipped sign or a missing 1/n would show up here immediately.
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        y = (X @ np.array([1.5, -2.0, 0.5]) > 0).astype(int)

        model = LogisticRegression(lr=0.1, n_iters=500, random_state=0).fit(X, y)

        losses = np.array(model.loss_history_)
        assert np.all(np.diff(losses) <= 1e-12)

    def test_gradient_matches_finite_differences(self):
        """The hand-derived gradient should agree with a numerical estimate of it."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        y = (rng.random(20) > 0.5).astype(float)
        w = rng.normal(size=3)
        b = 0.4
        eps = 1e-6

        z = X @ w + b
        analytic = X.T @ (1.0 / (1.0 + np.exp(-z)) - y) / len(y)
        numeric = np.array(
            [
                (log_loss(X, y, w + eps * e, b) - log_loss(X, y, w - eps * e, b))
                / (2 * eps)
                for e in np.eye(3)
            ]
        )

        np.testing.assert_allclose(analytic, numeric, atol=1e-6)

    def test_l2_shrinks_the_weights(self):
        """Regularization is meant to pull the weights towards zero."""
        X, y = make_blobs(n_samples=100, centers=2, random_state=42)
        X = standardize(X)

        plain = LogisticRegression(lr=0.1, n_iters=1000, lam=0.0, random_state=0)
        ridged = LogisticRegression(lr=0.1, n_iters=1000, lam=1.0, random_state=0)

        norm_plain = np.linalg.norm(plain.fit(X, y).w_)
        norm_ridged = np.linalg.norm(ridged.fit(X, y).w_)

        assert norm_ridged < norm_plain


class TestAgainstReferences:
    """Checks against scikit-learn and against the pure-Python version."""

    def test_matches_sklearn(self):
        """Same decisions as scikit-learn on overlapping, non-separable classes.

        Coefficients are compared loosely on purpose: gradient descent and L-BFGS take
        different paths to the same optimum, and on well-separated data the optimum is
        at infinity, so what has to agree is the boundary, not the raw numbers.
        """
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression as SkLogisticRegression

        X, y = make_blobs(n_samples=200, centers=2, cluster_std=3.0, random_state=0)
        X = standardize(X)

        ours = LogisticRegression(lr=0.5, n_iters=5000, lam=0.0, random_state=0)
        ours.fit(X, y)
        theirs = SkLogisticRegression(penalty=None, max_iter=5000).fit(X, y)

        agreement = np.mean(ours.predict(X) == theirs.predict(X))
        assert agreement >= 0.95

        our_accuracy = accuracy_score(y, ours.predict(X))
        their_accuracy = accuracy_score(y, theirs.predict(X))
        assert our_accuracy == pytest.approx(their_accuracy, abs=0.02)

    def test_scratch_and_numpy_agree(self):
        """Both minimise the same loss, so they land in the same place."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 2))
        y = (X @ np.array([1.0, -1.0]) + rng.normal(scale=0.5, size=40) > 0).astype(int)

        numpy_model = LogisticRegression(lr=0.5, n_iters=4000, random_state=0)
        numpy_model.fit(X, y)

        scratch_model = LogisticRegressionScratch(lr=0.5, n_iters=4000, random_state=0)
        scratch_model.fit(X.tolist(), y.tolist())

        np.testing.assert_allclose(
            scratch_model.predict_proba(X.tolist()),
            numpy_model.predict_proba(X),
            atol=0.02,
        )


class TestEdgeCases:
    """The inputs that actually break things."""

    def test_predict_before_fit_raises(self):
        """An unfitted model must say so, not crash on `X @ None`."""
        model = LogisticRegression()
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((2, 2)))

    def test_rejects_non_binary_labels(self):
        """Labels {1, 2} would otherwise train a silently meaningless model."""
        X = np.array([[0.0], [1.0], [2.0]])
        with pytest.raises(ValueError, match="exactly the labels 0 and 1"):
            LogisticRegression().fit(X, np.array([1, 2, 1]))

    def test_rejects_a_single_class(self):
        """Nothing to separate, so fitting is a user error rather than a no-op."""
        X = np.array([[0.0], [1.0], [2.0]])
        with pytest.raises(ValueError, match="exactly the labels 0 and 1"):
            LogisticRegression().fit(X, np.array([1, 1, 1]))

    def test_rejects_mismatched_lengths(self):
        """More rows of X than labels."""
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            LogisticRegression().fit(np.zeros((3, 2)), np.array([0, 1]))

    def test_rejects_1d_X(self):
        """A flat array is ambiguous, it could be one feature or one sample."""
        with pytest.raises(ValueError, match="2-dimensional"):
            LogisticRegression().fit(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))

    def test_extreme_inputs_do_not_overflow(self):
        """|z| in the thousands must give a finite probability, no overflow."""
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression(lr=0.5, n_iters=500, random_state=42).fit(X, y)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            probs = model.predict_proba(np.array([[1e6], [-1e6]]))

        assert np.all(np.isfinite(probs))
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_single_feature_and_single_sample(self):
        """The smallest thing that can still be called training."""
        model = LogisticRegression(lr=0.5, n_iters=100, random_state=0)
        model.fit(np.array([[-1.0], [1.0]]), np.array([0, 1]))

        assert model.predict(np.array([[5.0]])).shape == (1,)

    def test_is_reproducible(self):
        """Same seed, same weights. This is what `random_state` is for."""
        X, y = make_blobs(n_samples=60, centers=2, random_state=1)
        X = standardize(X)

        first = LogisticRegression(random_state=7).fit(X, y)
        second = LogisticRegression(random_state=7).fit(X, y)

        np.testing.assert_array_equal(first.w_, second.w_)


def test_viz_panel_produces_frames():
    """The panel is data until something draws it, so it tests without a browser."""
    pytest.importorskip("matplotlib")
    from mlblueprint.linear.logistic_regression.viz import LogisticRegressionPanel

    panel = LogisticRegressionPanel()
    frames = panel.frames(lr=0.1, n_iters=50)

    assert len(frames) > 1
    assert all(f.caption for f in frames)
    assert frames[-1].metrics["loss"] < frames[0].metrics["loss"]


class TestHandWorked:
    """Exact answers computable on paper, no randomness involved."""

    def test_zero_weights_give_half_probability(self):
        """With w=0 and b=0, z=0 everywhere, so sigmoid(0) is exactly 0.5."""
        model = LogisticRegression()
        model.w_ = np.array([0.0])
        model.b_ = 0.0
        model.loss_history_ = [float(np.log(2))]

        np.testing.assert_allclose(
            model.predict_proba(np.array([[0.0], [5.0], [-5.0]])),
            np.array([0.5, 0.5, 0.5]),
        )
        np.testing.assert_array_equal(
            model.predict(np.array([[-5.0], [5.0]])), np.array([1, 1])
        )

    def test_hand_set_boundary_decides_by_sign(self):
        """With w=[1] and b=0, z=x, so negative x is class 0, positive is 1."""
        model = LogisticRegression()
        model.w_ = np.array([1.0])
        model.b_ = 0.0
        model.loss_history_ = [0.5]

        np.testing.assert_array_equal(
            model.predict(np.array([[-2.0], [2.0]])), np.array([0, 1])
        )
        assert model.predict_proba(np.array([[0.0]]))[0] == pytest.approx(0.5)


class TestScratchParity:
    """The pure-Python version must match the NumPy version's contract."""

    def test_scratch_loss_decreases(self):
        """Same convex downhill guarantee as the NumPy version."""
        X = [[-2.0], [-1.0], [1.0], [2.0]]
        y = [0, 0, 1, 1]
        model = LogisticRegressionScratch(lr=0.5, n_iters=500, random_state=42)
        model.fit(X, y)

        assert model.loss_history_[-1] < model.loss_history_[0]
        assert model.predict(X) == y

    def test_scratch_extreme_inputs_do_not_overflow(self):
        """Large |z| must give a finite probability, no overflow or crash."""
        import math

        assert LogisticRegressionScratch is not None
        model = LogisticRegressionScratch(lr=0.5, n_iters=200, random_state=0)
        model.fit([[-2.0], [2.0]], [0, 1])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            probs = model.predict_proba([[1000.0], [-1000.0]])

        assert all(math.isfinite(p) for p in probs)
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert probs[0] > 0.99 and probs[1] < 0.01

    def test_scratch_rejects_bad_labels(self):
        """Scratch validates labels exactly like the NumPy version."""
        with pytest.raises(ValueError, match="exactly the labels 0 and 1"):
            LogisticRegressionScratch().fit([[0.0], [1.0]], [1, 2])
        with pytest.raises(ValueError, match="exactly the labels 0 and 1"):
            LogisticRegressionScratch().fit([[0.0], [1.0]], [1, 1])

    def test_scratch_predict_before_fit_raises(self):
        """Unfitted scratch model must say so via NotFittedError."""
        with pytest.raises(NotFittedError):
            LogisticRegressionScratch().predict_proba([[0.0]])

    def test_scratch_is_reproducible(self):
        """Same seed, same weights — the scratch random_state contract."""
        X = [[-1.0], [1.0], [-0.5], [0.5]]
        y = [0, 1, 0, 1]
        first = LogisticRegressionScratch(random_state=7).fit(X, y)
        second = LogisticRegressionScratch(random_state=7).fit(X, y)

        assert first.w_ == second.w_
        assert first.loss_history_ == second.loss_history_

    def test_rejects_empty_input(self):
        """Empty input is a user error, not a silent no-op."""
        with pytest.raises(ValueError, match="empty|at least one sample"):
            LogisticRegression().fit(np.zeros((0, 2)), np.array([]))
