"""Visualisation panel: the decision boundary as gradient descent moves it."""

from __future__ import annotations

import numpy as np

from mlblueprint.core.datasets import make_blobs
from mlblueprint.core.metrics import accuracy_score
from mlblueprint.core.viz import Frame, Parameter, VizPanel, register_panel

from .numpy_impl import LogisticRegression

SEED = 42


@register_panel
class LogisticRegressionPanel(VizPanel):
    """Boundary and probability field over the course of training."""

    name = "Logistic Regression — the boundary settling"
    family = "linear"
    description = "Watch the decision boundary rotate into place as the loss falls."

    def parameters(self) -> list[Parameter]:
        """Learning rate, iteration budget and how much the classes overlap."""
        return [
            Parameter(
                "lr",
                "Learning rate",
                default=0.1,
                kind="float",
                min=0.001,
                max=5.0,
                step=0.001,
                help="Step size. Crank it up and watch the boundary overshoot.",
            ),
            Parameter(
                "n_iters",
                "Iterations",
                default=200,
                kind="int",
                min=1,
                max=2000,
                help="How long to train. The last frame is the fitted model.",
            ),
            Parameter(
                "cluster_std",
                "Class overlap",
                default=1.5,
                kind="float",
                min=0.5,
                max=5.0,
                step=0.1,
                help="Spread of each blob. Large values make the classes inseparable.",
            ),
        ]

    def frames(
        self, lr: float = 0.1, n_iters: int = 200, cluster_std: float = 1.5
    ) -> list[Frame]:
        """Return one frame per snapshot, log-spaced so the early motion is visible."""
        X, y = make_blobs(
            n_samples=200, centers=2, cluster_std=cluster_std, random_state=SEED
        )
        # A single fixed learning rate only behaves when the features are comparable.
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        pad = 0.5
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 200),
            np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 200),
        )
        grid = np.column_stack([xx.ravel(), yy.ravel()])

        # Snapshots come from refitting with a shorter budget rather than from
        # reaching into the training loop. Slower than recording one run, but
        # far easier to read — and with 200 points and <=2000 iters it still
        # finishes in about a second. Log spacing because almost all of the
        # movement happens in the first few dozen iterations.
        checkpoints = sorted({max(1, round(n_iters ** (i / 19))) for i in range(20)})

        frames = []
        for stop in checkpoints:
            model = LogisticRegression(lr=lr, n_iters=stop, random_state=SEED)
            model.fit(X, y)

            probs = model.predict_proba(grid).reshape(xx.shape)
            loss = model.loss_history_[-1]
            accuracy = accuracy_score(y, model.predict(X))

            # Bind the per-frame array as a default arg so every frame keeps
            # its own step (plain closure would show the last step 20 times).
            # xx/yy/X/y are loop-invariant, so they can stay as closures.
            def draw(ax, probs=probs):
                ax.contourf(xx, yy, probs, levels=20, cmap="RdBu", alpha=0.6)
                ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)
                ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="black", s=25)
                ax.set_xlabel("feature 1")
                ax.set_ylabel("feature 2")

            frames.append(
                Frame(
                    draw=draw,
                    caption=f"After {stop} iteration{'s' if stop > 1 else ''}",
                    metrics={"loss": round(loss, 4), "accuracy": round(accuracy, 3)},
                )
            )

        return frames
