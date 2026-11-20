"""End-to-end run of every stage at tiny sizes against a temporary sqlite MLflow store."""

import json
import os
import shutil
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pytest

import mlflow
from agent_platform.sim.episode import run_episode
from agent_platform.training import pipeline, promote
from agent_platform.training.config import PipelineConfig
from agent_platform.training.registry import Candidate, select_champion
from agent_platform.training.tracking import RunArtifactDatasetSource
from mlflow import MlflowClient

REPO_DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    data_dir = root / "data"
    shutil.copytree(REPO_DATA / "flywire", data_dir / "flywire")
    previous_uri, previous_cwd = mlflow.get_tracking_uri(), Path.cwd()
    os.chdir(root)  # the default artifact root is ./mlruns
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    cfg = PipelineConfig(
        data_dir=data_dir,
        n_episodes=24,
        n_eval_episodes=6,
        n_steps=(2,),
        eda_max_steps=3,
        eda_rows=400,
        success_gate_margin=1.0,  # tiny data gives poor models; this test checks the mechanics
    )
    try:
        pipeline.run(cfg, pipeline.STAGES)
        yield cfg, MlflowClient()
    finally:
        os.chdir(previous_cwd)
        mlflow.set_tracking_uri(previous_uri)


def runs_of(client, experiment_name):
    exp = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs([exp.experiment_id])
    return {r.data.tags.get("mlflow.runName"): r for r in runs}


def inputs_by_context(run):
    result = {}
    for item in run.inputs.dataset_inputs:
        tags = {t.key: t.value for t in item.tags}
        result[tags["mlflow.data.context"]] = item.dataset
    return result


def test_training_runs_compared_in_one_experiment(env):
    cfg, client = env
    runs = runs_of(client, cfg.experiment)
    assert set(runs) == {"baseline", "baseline-gbdt", "flywire-k2"}
    for run in runs.values():
        assert run.info.status == "FINISHED"
        for key in ("test_f1_macro", "test_accuracy", "episode_success_rate", "train_f1_macro"):
            assert key in run.data.metrics
        assert "predict_latency_ms" in run.data.metrics
        assert run.data.params["variant"] in runs
    assert runs["flywire-k2"].data.params["connectome.n_steps"] == "2"


def test_datasets_are_linked_to_runs_with_lineage(env):
    cfg, client = env
    gen_run = runs_of(client, cfg.data_experiment)["generate-dataset"]
    runs = runs_of(client, cfg.experiment)
    digests = set()
    for name, run in runs.items():
        by_context = inputs_by_context(run)
        assert by_context["training"].name == "fly-bc-train"
        assert by_context["evaluation"].name == "fly-bc-test"
        source = json.loads(by_context["training"].source)
        assert source["run_id"] == gen_run.info.run_id
        assert by_context["training"].source_type == "mlflow_run_artifact"
        digests.add(by_context["training"].digest)
        assert ("connectome" in by_context) == name.startswith("flywire")
    assert len(digests) == 1  # every run trained on the identical data
    fly = inputs_by_context(runs["flywire-k2"])["connectome"]
    assert fly.name == "flywire-v783-subgraph"
    assert fly.source_type == "http"
    assert "91bdd1e7" in json.loads(fly.source)["url"]
    # The dataset source really resolves to the stored file.
    path = RunArtifactDatasetSource.from_dict(json.loads(by_context_source(runs))).load()
    assert Path(path).name == "train.parquet"


def by_context_source(runs):
    return inputs_by_context(runs["baseline"])["training"].source


def test_generation_and_eda_runs_have_artifacts(env):
    cfg, client = env
    data_runs = runs_of(client, cfg.data_experiment)
    gen = data_runs["generate-dataset"]
    names = {a.path for a in client.list_artifacts(gen.info.run_id, "datasets")}
    assert {"datasets/train.parquet", "datasets/test.parquet", "datasets/generation.json"} <= names
    assert gen.data.params["gen.n_episodes"] == "24"

    eda = data_runs["eda"]
    names = {a.path for a in client.list_artifacts(eda.info.run_id, "eda")}
    assert "eda/class_balance.png" in names
    assert "eda/flywire_propagation.png" in names
    assert "eda/eda_summary.md" in names
    assert "class_imbalance_ratio" in eda.data.metrics
    assert set(inputs_by_context(eda)) == {"eda"}
    local = client.download_artifacts(eda.info.run_id, "eda/eda_summary.json")
    summary = json.loads(Path(local).read_text(encoding="utf-8"))
    assert summary["flywire_graph"]["neurons"] >= 100


def test_training_run_diagnostic_artifacts(env):
    cfg, client = env
    run = runs_of(client, cfg.experiment)["flywire-k2"]
    names = {a.path for a in client.list_artifacts(run.info.run_id, "diagnostics")}
    assert {
        "diagnostics/confusion_matrix.png",
        "diagnostics/confusion_matrix.csv",
        "diagnostics/classification_report.json",
        "diagnostics/trajectories_model_vs_teacher.png",
        "diagnostics/episode_results.csv",
    } <= names


def test_registry_versions_alias_and_champion_model(env):
    cfg, client = env
    runs = runs_of(client, cfg.experiment)
    versions = client.search_model_versions(f"name = '{cfg.model_name}'")
    assert len(versions) == 3
    assert {v.run_id for v in versions} == {r.info.run_id for r in runs.values()}
    for v in versions:
        assert v.tags["variant"] in runs
        assert "test_f1_macro" in v.tags
        assert "dataset_digest_train" in v.tags
        assert v.description

    champion = client.get_model_version_by_alias(cfg.model_name, cfg.alias)
    by_id = {r.info.run_id: r for r in runs.values()}
    candidates = [
        Candidate(
            int(v.version),
            v.tags["variant"],
            v.run_id,
            by_id[v.run_id].data.metrics["test_f1_macro"],
            by_id[v.run_id].data.metrics["episode_success_rate"],
            by_id[v.run_id].data.metrics["teacher_success_rate"],
        )
        for v in versions
    ]
    expected = select_champion(candidates, cfg.success_gate_margin)
    assert int(champion.version) == expected.version

    model = mlflow.sklearn.load_model(f"models:/{cfg.model_name}@{cfg.alias}")
    result = run_episode(lambda obs: int(model.predict(obs.reshape(1, -1))[0]), seed=5)
    assert result.steps > 0 and len(result.trajectory) == result.steps + 1
    assert model.predict(np.zeros((2, 9))).dtype.kind == "i"


def test_rerun_registration_and_alias_switch_cli(env, capsys):
    cfg, client = env
    pipeline.run(cfg, ["register"])  # safe to repeat: new versions, nothing crashes
    assert len(client.search_model_versions(f"name = '{cfg.model_name}'")) == 6

    promote.main(["--model-name", cfg.model_name, "--version", "1", "--alias", cfg.alias])
    out = capsys.readouterr().out
    assert int(client.get_model_version_by_alias(cfg.model_name, cfg.alias).version) == 1
    assert "champion" in out and f"registered model: {cfg.model_name} (6 versions)" in out

    promote.main(["--model-name", cfg.model_name])  # no --version: only prints
    assert "fly-policy" in capsys.readouterr().out
    with pytest.raises(mlflow.exceptions.MlflowException):
        promote.main(["--model-name", cfg.model_name, "--version", "99"])


def candidate(version, f1, success, teacher=0.9):
    return Candidate(version, f"v{version}", f"run{version}", f1, success, teacher)


def test_select_champion_uses_f1_with_success_gate():
    cands = [candidate(1, 0.6, 0.9), candidate(2, 0.95, 0.5), candidate(3, 0.9, 0.85)]
    assert select_champion(cands, margin=0.1).version == 3  # v2 fails the gate
    assert select_champion(cands, margin=1.0).version == 2


def test_select_champion_ties_and_empty():
    tied = [candidate(1, 0.9, 0.8), candidate(2, 0.9, 0.9)]
    assert select_champion(tied, margin=0.5).version == 2
    with pytest.raises(RuntimeError, match="gate"):
        select_champion([candidate(1, 0.9, 0.1)], margin=0.1)


def test_flywire_stage_verifies_and_detects_tampering(tmp_path):
    shutil.copytree(REPO_DATA / "flywire", tmp_path / "flywire")
    cfg = PipelineConfig(data_dir=tmp_path)
    pipeline.stage_flywire(cfg, rebuild=False)
    edges = tmp_path / "flywire" / "subgraph_edges.parquet"
    edges.write_bytes(edges.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="does not match"):
        pipeline.stage_flywire(cfg, rebuild=False)


def test_parser_defaults():
    args = pipeline.build_parser().parse_args([])
    assert args.stage == list(pipeline.STAGES)
    assert args.alias == "champion" and args.model_name == "fly-policy"


def test_main_builds_config_from_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        pipeline, "run", lambda cfg, stages, rebuild: seen.update(cfg=cfg, stages=stages)
    )
    pipeline.main(
        ["--stage", "eda", "train", "--episodes", "10", "--n-steps", "3", "--threads", "2"]
    )
    assert seen["stages"] == ["eda", "train"]
    assert (seen["cfg"].n_episodes, seen["cfg"].n_steps, seen["cfg"].threads) == (10, (3,), 2)
