"""Worked end-to-end run of Logistic Regression on two synthetic blobs."""

from mlblueprint.core.datasets import make_blobs
from mlblueprint.core.metrics import accuracy_score
from mlblueprint.core.random import DEFAULT_SEED, check_random_state
from mlblueprint.linear.logistic_regression import LogisticRegression


def main():
    X, y = make_blobs(
        n_samples=200, centers=2, cluster_std=1.5, random_state=DEFAULT_SEED
    )

    # Centre and scale. A single fixed learning rate only works when every feature
    # is on a comparable scale, otherwise one big column dictates the step size.
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    rng = check_random_state(DEFAULT_SEED)
    order = rng.permutation(len(X))
    split = int(0.75 * len(X))
    train, test = order[:split], order[split:]

    model = LogisticRegression(lr=0.1, n_iters=1000, lam=0.0, random_state=DEFAULT_SEED)
    model.fit(X[train], y[train])

    print(model)
    first, last = model.loss_history_[0], model.loss_history_[-1]
    print(f"loss            {first:.4f} -> {last:.4f}")
    print(f"train accuracy  {accuracy_score(y[train], model.predict(X[train])):.3f}")
    print(f"test accuracy   {accuracy_score(y[test], model.predict(X[test])):.3f}")

    print("\nfirst five test probabilities:")
    for prob, true in zip(model.predict_proba(X[test])[:5], y[test][:5], strict=True):
        print(f"  P(y=1) = {prob:.3f}   actual = {true}")


if __name__ == "__main__":
    main()
