import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_platform.sim.env import OBS_DIM
from agent_platform.training import dataset as ds
from agent_platform.training import flywire
from agent_platform.training.dataset import GenerationParams

REPO_DATA = Path(__file__).resolve().parents[2] / "data"


def raw_table() -> pd.DataFrame:
    """Tiny table in the layout of Connectivity_783.parquet: a 4-ring plus a dangling tail."""
    pairs = [(0, 1, 5), (1, 2, -4), (2, 3, 6), (3, 0, 7), (3, 4, 1), (4, 5, 1)]
    return pd.DataFrame(
        {
            "Presynaptic_ID": [1000 + a for a, _, _ in pairs],
            "Postsynaptic_ID": [1000 + b for _, b, _ in pairs],
            "Presynaptic_Index": [a for a, _, _ in pairs],
            "Postsynaptic_Index": [b for _, b, _ in pairs],
            "Connectivity": [abs(w) for _, _, w in pairs],
            "Excitatory": [np.sign(w) for _, _, w in pairs],
            "Excitatory x Connectivity": [w for _, _, w in pairs],
        }
    )


def test_select_subgraph_keeps_strongly_connected_hubs():
    edges, neurons = flywire.select_subgraph(raw_table(), n_neurons=5)
    assert len(neurons) == 4  # the tail neurons are not strongly connected to the ring
    assert set(neurons["root_id"]) == {1000, 1001, 1002, 1003}
    assert len(edges) == 4
    assert set(edges["weight"]) == {5, -4, 6, 7}
    assert edges[["pre", "post"]].max().max() == 3


def test_download_and_build_subgraph(tmp_path, monkeypatch):
    payload = io.BytesIO()
    raw_table().to_parquet(payload)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        flywire.urllib.request, "urlopen", lambda url, timeout: Response(payload.getvalue())
    )
    meta = flywire.build_subgraph(tmp_path, n_neurons=5)
    assert meta["source_url"] == flywire.RAW_URL
    assert flywire.COMMIT in meta["source_url"]
    assert meta["raw_file_sha256"] == flywire.sha256_of(tmp_path / "raw" / flywire.RAW_FILENAME)
    assert meta["counts"]["neurons"] == 4
    assert flywire.load_metadata(tmp_path) == meta
    # A second build reuses the downloaded file instead of fetching it again.
    monkeypatch.setattr(flywire.urllib.request, "urlopen", lambda *a, **k: 1 / 0)
    assert (
        flywire.build_subgraph(tmp_path, n_neurons=5)["raw_file_sha256"] == meta["raw_file_sha256"]
    )


def test_committed_subgraph_matches_its_metadata():
    edges_path, neurons_path, meta_path = flywire.subgraph_paths(REPO_DATA)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    edges = pd.read_parquet(edges_path)
    neurons = pd.read_parquet(neurons_path)
    assert meta["flywire_version"] == 783
    assert meta["repository_commit"] == flywire.COMMIT
    assert len(meta["raw_file_sha256"]) == 64
    assert meta["counts"]["neurons"] == len(neurons) >= 100
    assert meta["counts"]["edges"] == len(edges)
    assert meta["files"][flywire.EDGES_FILE] == flywire.sha256_of(edges_path)
    assert edges_path.stat().st_size < 500 * 1024
    assert (edges["weight"] != 0).all()
    assert edges["pre"].max() < len(neurons) and edges["post"].max() < len(neurons)


def test_generation_is_reproducible_and_split_by_episode(tmp_path):
    params = GenerationParams(n_episodes=24, n_eval_episodes=4)
    meta = ds.generate(tmp_path / "a", params)
    ds.generate(tmp_path / "b", params)
    train, test = ds.load_split(tmp_path / "a")
    train_b, _ = ds.load_split(tmp_path / "b")
    pd.testing.assert_frame_equal(train, train_b)

    assert list(train.columns) == ["episode_seed", "step", *ds.FEATURE_NAMES, ds.TARGET]
    assert len(ds.FEATURE_NAMES) == OBS_DIM
    assert set(train["episode_seed"]).isdisjoint(set(test["episode_seed"]))
    assert len(set(train["episode_seed"]) | set(test["episode_seed"])) == 24
    assert set(train[ds.TARGET]) <= {0, 1, 2}
    # Steps within an episode are consecutive from zero.
    first = train[train["episode_seed"] == train["episode_seed"].iloc[0]]
    assert first["step"].tolist() == list(range(len(first)))
    assert meta["generation"]["n_episodes"] == 24
    assert ds.load_metadata(tmp_path / "a") == meta
    assert meta["env_version"] and meta["teacher"]
    assert (tmp_path / "a" / ds.EPISODES_FILE).exists()


def test_split_fraction_and_minimum():
    frame = pd.DataFrame({"episode_seed": np.repeat(np.arange(8), 3), "v": np.arange(24)})
    train, test = ds.split_by_episode(frame, 0.25, 0)
    assert test["episode_seed"].nunique() == 2
    train2, test2 = ds.split_by_episode(frame, 0.0, 0)
    assert test2["episode_seed"].nunique() == 1  # never an empty test set
    assert len(train) + len(test) == len(train2) + len(test2) == 24


def test_eval_seeds_do_not_collide_with_dataset_seeds():
    p = GenerationParams()
    assert p.seed_start + p.n_episodes <= p.eval_seed_start


@pytest.mark.parametrize("alias", ["champion", "other"])
def test_config_env_overrides(monkeypatch, alias):
    from agent_platform.training.config import PipelineConfig

    monkeypatch.setenv("FLY_MODEL_ALIAS", alias)
    monkeypatch.setenv("FLY_DATA_DIR", "somewhere")
    cfg = PipelineConfig()
    assert cfg.alias == alias
    assert cfg.data_dir == Path("somewhere")
    assert cfg.work_dir == Path("somewhere") / "generated"
