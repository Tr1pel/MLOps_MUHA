"""Fixed connectome layer: a scikit-learn transformer around a FlyWire subgraph."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


def load_edges(path: str | Path) -> pd.DataFrame:
    """Edge table with columns pre, post (0-based neuron indices) and signed integer weight."""
    return pd.read_parquet(path, columns=["pre", "post", "weight"])


def build_weight_matrix(edges: pd.DataFrame, spectral_radius: float) -> np.ndarray:
    """Dense signed matrix W[pre, post], rescaled so that max |eigenvalue| == spectral_radius."""
    n = int(max(edges["pre"].max(), edges["post"].max())) + 1
    w = np.zeros((n, n), dtype=np.float64)
    w[edges["pre"].to_numpy(), edges["post"].to_numpy()] = edges["weight"].to_numpy(np.float64)
    radius = float(np.max(np.abs(np.linalg.eigvals(w))))
    if radius <= 0.0:
        raise ValueError("connectome weight matrix has zero spectral radius")
    return w * (spectral_radius / radius)


class ConnectomeFeatures(BaseEstimator, TransformerMixin):
    """Observation -> activity of the connectome neurons after ``n_steps`` propagation steps.

    Everything is derived from the constructor parameters and the edge file; nothing is
    learned from the data. For every row independently:

        a_0 = 0;   a_t = tanh(a_{t-1} @ W + I);   output a_{n_steps}

    where W is the signed synapse-count matrix rescaled to a fixed spectral radius and I
    injects ``obs @ P`` into ``n_input`` randomly chosen (seeded) input neurons. tanh keeps the
    activity in (-1, 1); there is no memory between rows.
    """

    def __init__(
        self,
        graph_path: str | None = None,
        n_steps: int = 4,
        n_input: int = 64,
        input_scale: float = 1.5,
        spectral_radius: float = 0.9,
        seed: int = 0,
    ) -> None:
        self.graph_path = graph_path
        self.n_steps = n_steps
        self.n_input = n_input
        self.input_scale = input_scale
        self.spectral_radius = spectral_radius
        self.seed = seed

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float64)
        if self.graph_path is None:
            raise ValueError("graph_path is required")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.weights_ = build_weight_matrix(load_edges(self.graph_path), self.spectral_radius)
        n_neurons = self.weights_.shape[0]
        if not 1 <= self.n_input <= n_neurons:
            raise ValueError(f"n_input must be in [1, {n_neurons}]")
        rng = np.random.default_rng(self.seed)
        self.input_neurons_ = np.sort(rng.choice(n_neurons, size=self.n_input, replace=False))
        n_features = X.shape[1]
        self.projection_ = (
            rng.normal(size=(n_features, self.n_input)) / np.sqrt(n_features) * self.input_scale
        )
        self.n_features_in_ = n_features
        return self

    def propagate(self, X) -> np.ndarray:
        """Activity after every step, shape (n_steps, n_rows, n_neurons); also used by the EDA."""
        check_is_fitted(self, "weights_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        current = np.zeros((X.shape[0], self.weights_.shape[0]))
        current[:, self.input_neurons_] = X @ self.projection_
        activity = np.zeros_like(current)
        history = []
        for _ in range(self.n_steps):
            activity = np.tanh(activity @ self.weights_ + current)
            history.append(activity)
        return np.stack(history)

    def transform(self, X):
        check_is_fitted(self, "weights_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        current = np.zeros((X.shape[0], self.weights_.shape[0]))
        current[:, self.input_neurons_] = X @ self.projection_
        activity = np.zeros_like(current)
        for _ in range(self.n_steps):
            activity = np.tanh(activity @ self.weights_ + current)
        return activity
