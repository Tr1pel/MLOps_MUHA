import numpy as np
import pytest
from conftest import FakePolicy

import agent_platform.ml.inference as inference
from agent_platform.core.config import Settings
from agent_platform.ml.inference import (
    ModelLoadError,
    _redact,
    load_registered_model,
)
from agent_platform.sim.env import OBS_DIM


class Version:
    def __init__(self, version, run_id="run-1", source="s3://bucket/path"):
        self.version = version
        self.run_id = run_id
        self.source = source


class FakeClient:
    def __init__(self, events, versions, error=None):
        self.events = events
        self.versions = versions
        self.error = error

    def get_model_version_by_alias(self, name, alias):
        self.events.append(("resolve", name, alias))
        if self.error:
            raise self.error
        return Version(self.versions.pop(0))


@pytest.fixture
def settings():
    return Settings(_env_file=None, mlflow_tracking_uri="http://mlflow.test:5000")


@pytest.fixture
def events(monkeypatch):
    log = []

    def fake_load(tracking_uri, model_uri, dst_path):
        log.append(("load", model_uri))
        return FakePolicy()

    monkeypatch.setattr(inference, "_load_sklearn", fake_load)
    return log


def use_client(monkeypatch, events, versions, error=None):
    monkeypatch.setattr(
        inference, "_make_client", lambda uri: FakeClient(events, versions, error=error)
    )


def raising(exc):
    def loader(settings):
        raise exc

    return loader


def test_alias_is_resolved_first_then_exact_version_is_loaded(settings, events, monkeypatch):
    use_client(monkeypatch, events, ["7", "8"])  # alias would move to 8 on a second lookup
    loaded = load_registered_model(settings)
    assert events == [("resolve", "fly-policy", "champion"), ("load", "models:/fly-policy/7")]
    assert loaded.info.version == "7"
    assert loaded.info.run_id == "run-1"
    assert loaded.info.alias == "champion"
    assert loaded.info.load_duration_ms >= 0


def test_alias_not_found_is_categorised(settings, events, monkeypatch):
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

    error = MlflowException("alias missing", error_code=RESOURCE_DOES_NOT_EXIST)
    use_client(monkeypatch, events, [], error=error)
    with pytest.raises(ModelLoadError) as raised:
        load_registered_model(settings)
    assert raised.value.reason == "alias_not_found"
    assert not any(kind == "load" for kind, *_ in events)


def test_other_registry_errors_propagate(settings, events, monkeypatch):
    use_client(monkeypatch, events, [], error=ConnectionError("down"))
    with pytest.raises(ConnectionError):
        load_registered_model(settings)


def test_loader_requires_tracking_uri():
    with pytest.raises(ModelLoadError) as raised:
        load_registered_model(Settings(_env_file=None))
    assert raised.value.reason == "not_configured"


def test_model_with_wrong_output_shape_is_rejected(settings, monkeypatch):
    class Bad:
        def predict(self, X):
            assert X.shape == (1, OBS_DIM)
            return np.zeros((1, 3))

    use_client(monkeypatch, [], ["1"])
    monkeypatch.setattr(inference, "_load_sklearn", lambda *args: Bad())
    with pytest.raises(ModelLoadError) as raised:
        load_registered_model(settings)
    assert raised.value.reason == "load_failed"


def test_missing_run_id_and_credentials_in_source(settings, events, monkeypatch):
    class Client:
        def get_model_version_by_alias(self, name, alias):
            return Version("2", run_id="", source="https://u:pw@host/x")

    monkeypatch.setattr(inference, "_make_client", lambda uri: Client())
    info = load_registered_model(settings).info
    assert info.run_id is None
    assert info.source == "https://host/x"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", ""), ("s3://b/k", "s3://b/k"), ("http://a:b@h:1/p", "http://h:1/p")],
)
def test_redact(raw, expected):
    assert _redact(raw) == expected


def test_blank_tracking_uri_means_unset():
    assert Settings(_env_file=None, mlflow_tracking_uri="  ").mlflow_tracking_uri is None
    assert Settings(_env_file=None, mlflow_tracking_uri="http://m").mlflow_tracking_uri


def test_real_seams_delegate_to_mlflow(monkeypatch, tmp_path):
    import mlflow.sklearn

    import mlflow

    seen = {}
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: seen.update(tracking=uri))
    monkeypatch.setattr(mlflow, "set_registry_uri", lambda uri: seen.update(registry=uri))

    def fake_load(uri, dst_path):
        seen.update(uri=uri, dst=dst_path)
        return "model"

    monkeypatch.setattr(mlflow.sklearn, "load_model", fake_load)
    assert inference._load_sklearn("http://m:5000", "models:/n/1", str(tmp_path)) == "model"
    assert seen == {
        "tracking": "http://m:5000",
        "registry": "http://m:5000",
        "uri": "models:/n/1",
        "dst": str(tmp_path),
    }
    assert type(inference._make_client("http://m:5000")).__name__ == "MlflowClient"
