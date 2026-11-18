import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from agent_platform.ml.features import ConnectomeFeatures

N = 30


@pytest.fixture
def graph_path(tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    pre, post = np.nonzero(rng.random((N, N)) < 0.2)
    edges = pd.DataFrame(
        {
            "pre": pre.astype(np.int32),
            "post": post.astype(np.int32),
            "weight": rng.choice([-1, 1], size=len(pre)) * rng.integers(1, 50, size=len(pre)),
        }
    )
    path = tmp_path / "edges.parquet"
    edges.to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def x():
    return np.random.default_rng(1).uniform(-1, 1, size=(12, 9))


def test_shape_and_bounds(graph_path, x):
    out = ConnectomeFeatures(graph_path, n_steps=4, n_input=8).fit(x).transform(x)
    assert out.shape == (12, N)
    assert np.all(np.abs(out) < 1.0)


def test_deterministic_across_instances(graph_path, x):
    a = ConnectomeFeatures(graph_path, n_input=8, seed=3).fit(x).transform(x)
    b = ConnectomeFeatures(graph_path, n_input=8, seed=3).fit(x).transform(x)
    c = ConnectomeFeatures(graph_path, n_input=8, seed=4).fit(x).transform(x)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_rows_are_independent(graph_path, x):
    model = ConnectomeFeatures(graph_path, n_input=8).fit(x)
    full = model.transform(x)
    single = np.vstack([model.transform(row[None, :]) for row in x])
    np.testing.assert_allclose(full, single)
    np.testing.assert_allclose(model.transform(x[::-1]), full[::-1])


def test_fitted_weights_are_normalised(graph_path, x):
    model = ConnectomeFeatures(graph_path, spectral_radius=0.7, n_input=8).fit(x)
    radius = np.max(np.abs(np.linalg.eigvals(model.weights_)))
    assert radius == pytest.approx(0.7)


def test_propagate_history_matches_transform(graph_path, x):
    model = ConnectomeFeatures(graph_path, n_steps=5, n_input=8).fit(x)
    history = model.propagate(x)
    assert history.shape == (5, 12, N)
    np.testing.assert_array_equal(history[-1], model.transform(x))
    # Step 1 only excites the input neurons.
    silent = np.setdiff1d(np.arange(N), model.input_neurons_)
    assert np.all(history[0][:, silent] == 0.0)


def test_clone_and_pickle_roundtrip(graph_path, x):
    model = ConnectomeFeatures(graph_path, n_steps=3, n_input=8).fit(x)
    assert clone(model).get_params() == model.get_params()
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.transform(x), model.transform(x))


def test_pickled_pipeline_does_not_need_graph_file(graph_path, x):
    pipe = make_pipeline(ConnectomeFeatures(graph_path, n_input=8), LogisticRegression())
    pipe.fit(x, np.arange(12) % 3)
    payload = pickle.dumps(pipe)
    Path(graph_path).unlink()
    restored = pickle.loads(payload)
    np.testing.assert_array_equal(restored.predict(x), pipe.predict(x))


def test_validation_errors(graph_path, x):
    with pytest.raises(ValueError, match="graph_path"):
        ConnectomeFeatures().fit(x)
    with pytest.raises(ValueError, match="n_steps"):
        ConnectomeFeatures(graph_path, n_steps=0).fit(x)
    with pytest.raises(ValueError, match="n_input"):
        ConnectomeFeatures(graph_path, n_input=N + 1).fit(x)
    fitted = ConnectomeFeatures(graph_path, n_input=8).fit(x)
    with pytest.raises(ValueError, match="features"):
        fitted.transform(x[:, :5])


def test_zero_spectral_radius_rejected(tmp_path, x):
    path = tmp_path / "dag.parquet"
    pd.DataFrame({"pre": [0], "post": [1], "weight": [3]}).to_parquet(path)
    with pytest.raises(ValueError, match="spectral radius"):
        ConnectomeFeatures(str(path), n_input=1).fit(x)
