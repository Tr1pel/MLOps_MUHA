"""Training stage: one MLflow run per model variant, all on the same data and eval seeds."""

import logging
import math
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow.sklearn
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.models import infer_signature

import mlflow
from agent_platform.ml.features import load_edges
from agent_platform.sim.env import ACTIONS, ENV_VERSION
from agent_platform.sim.episode import EpisodeResult
from agent_platform.sim.teacher import Teacher
from agent_platform.training import dataset as ds
from agent_platform.training import flywire, tracking
from agent_platform.training.config import PipelineConfig
from agent_platform.training.evaluate import (
    episode_metrics,
    latency_ms,
    model_policy,
    offline_metrics,
    run_episodes,
)
from agent_platform.training.models import (
    VariantSpec,
    build_pipeline,
    describe_params,
    variant_specs,
)
from agent_platform.training.plots import confusion_figure, trajectory_grid

logger = logging.getLogger(__name__)

MODEL_ARTIFACT = "model"
SESSION_TAG = "pipeline.session"


@dataclass
class Session:
    cfg: PipelineConfig
    session_id: str
    graph_path: Path
    flywire_meta: dict
    gen_meta: dict
    train: pd.DataFrame
    test: pd.DataFrame
    train_dataset: PandasDataset
    test_dataset: PandasDataset
    flywire_dataset: PandasDataset
    eval_seeds: range
    teacher_results: list[EpisodeResult]


def git_commit() -> str:
    for env_name in ("GIT_COMMIT", "APP_COMMIT"):
        if os.environ.get(env_name):
            return os.environ[env_name]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def prepare_session(cfg: PipelineConfig) -> Session:
    gen_run_id = tracking.read_generation_run_id(cfg.work_dir)
    gen_meta = ds.load_metadata(cfg.work_dir)
    train, test = ds.load_split(cfg.work_dir)
    edges_path, _, _ = flywire.subgraph_paths(cfg.data_dir)
    params = ds.GenerationParams(**gen_meta["generation"])
    eval_seeds = range(params.eval_seed_start, params.eval_seed_start + cfg.n_eval_episodes)
    teacher_results = run_episodes(Teacher(), eval_seeds, cfg.max_steps)
    return Session(
        cfg=cfg,
        session_id=uuid.uuid4().hex[:8],
        graph_path=edges_path,
        flywire_meta=flywire.load_metadata(cfg.data_dir),
        gen_meta=gen_meta,
        train=train,
        test=test,
        train_dataset=tracking.generated_dataset(train, "train", gen_run_id),
        test_dataset=tracking.generated_dataset(test, "test", gen_run_id),
        flywire_dataset=tracking.flywire_dataset(load_edges(edges_path)),
        eval_seeds=eval_seeds,
        teacher_results=teacher_results,
    )


def _pip_requirements() -> list[str]:
    return [f"{pkg}=={version(pkg)}" for pkg in ("scikit-learn", "numpy", "scipy", "pandas")]


def _log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics({k: v for k, v in metrics.items() if not math.isnan(v)})


def _trajectory_seeds(model_results: list[EpisodeResult], n: int = 6) -> list[int]:
    """Model failures first (most informative), then the first seeds, without duplicates."""
    picked = [i for i, r in enumerate(model_results) if not r.success][: n // 2]
    picked += [i for i in range(len(model_results)) if i not in picked][: n - len(picked)]
    return sorted(picked)


def train_variant(spec: VariantSpec, session: Session) -> str:
    cfg = session.cfg
    features = list(ds.FEATURE_NAMES)
    x_train = session.train[features].to_numpy()
    y_train = session.train[ds.TARGET].to_numpy()
    x_test = session.test[features].to_numpy()
    y_test = session.test[ds.TARGET].to_numpy()

    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=spec.name) as run:
        pipeline = build_pipeline(spec, cfg, session.graph_path)
        mlflow.set_tags(
            {
                "stage": "train",
                "variant": spec.name,
                "git_commit": git_commit(),
                SESSION_TAG: session.session_id,
                "primary_metric": "test_f1_macro",
            }
        )
        mlflow.log_params(describe_params(spec, pipeline))
        mlflow.log_params(
            {
                "env_version": ENV_VERSION,
                "n_eval_episodes": cfg.n_eval_episodes,
                "eval_seed_start": session.eval_seeds.start,
                "max_steps": cfg.max_steps,
                "data.n_episodes": session.gen_meta["generation"]["n_episodes"],
                "data.split_seed": session.gen_meta["generation"]["split_seed"],
                "data.rows_train": len(session.train),
                "data.rows_test": len(session.test),
            }
        )
        mlflow.log_input(session.train_dataset, context="training")
        mlflow.log_input(session.test_dataset, context="evaluation")
        if spec.family == "connectome":
            mlflow.log_input(
                session.flywire_dataset,
                context="connectome",
                tags=tracking.flywire_input_tags(session.flywire_meta),
            )
            mlflow.log_params(
                {
                    "flywire.version": session.flywire_meta["flywire_version"],
                    "flywire.neurons": session.flywire_meta["counts"]["neurons"],
                    "flywire.edges": session.flywire_meta["counts"]["edges"],
                    "flywire.raw_sha256": session.flywire_meta["raw_file_sha256"],
                }
            )

        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        mlflow.log_metric("fit_seconds", time.perf_counter() - started)

        metrics, diag = offline_metrics(pipeline, x_train, y_train, x_test, y_test)
        results = run_episodes(model_policy(pipeline), session.eval_seeds, cfg.max_steps)
        metrics.update(episode_metrics(results))
        metrics.update(episode_metrics(session.teacher_results, prefix="teacher"))
        metrics["predict_latency_ms"] = latency_ms(pipeline, x_test)
        _log_metrics(metrics)

        _log_diagnostics(spec, diag, results, session)
        info = mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT,
            signature=infer_signature(x_test[:5], pipeline.predict(x_test[:5])),
            input_example=x_test[:5],
            pip_requirements=_pip_requirements(),
            serialization_format="cloudpickle",
        )
        mlflow.set_tag("model_id", info.model_id)
        logger.info(
            "%s: test_f1_macro=%.4f episode_success_rate=%.2f",
            spec.name,
            metrics["test_f1_macro"],
            metrics["episode_success_rate"],
        )
        return run.info.run_id


def _log_diagnostics(spec, diag, results, session: Session) -> None:
    cm = diag["confusion"]
    fig = confusion_figure(cm, f"{spec.name}: test confusion matrix")
    mlflow.log_figure(fig, "diagnostics/confusion_matrix.png")
    plt.close(fig)
    cm_table = pd.DataFrame(cm, index=list(ACTIONS), columns=list(ACTIONS))
    mlflow.log_text(cm_table.rename_axis("teacher").to_csv(), "diagnostics/confusion_matrix.csv")
    mlflow.log_dict(diag["report"], "diagnostics/classification_report.json")

    table = pd.DataFrame(
        {
            "seed": [r.seed for r in results],
            "success": [r.success for r in results],
            "steps": [r.steps for r in results],
            "collisions": [r.collisions for r in results],
            "teacher_success": [r.success for r in session.teacher_results],
            "teacher_steps": [r.steps for r in session.teacher_results],
        }
    )
    mlflow.log_text(table.to_csv(index=False), "diagnostics/episode_results.csv")

    picked = _trajectory_seeds(results)
    fig = trajectory_grid(
        [(session.teacher_results[i], results[i]) for i in picked],
        f"{spec.name}: closed-loop trajectories (model failures first)",
    )
    mlflow.log_figure(fig, "diagnostics/trajectories_model_vs_teacher.png")
    plt.close(fig)


def train_all(cfg: PipelineConfig) -> list[str]:
    session = prepare_session(cfg)
    mlflow.set_experiment(cfg.experiment)
    run_ids = [train_variant(spec, session) for spec in variant_specs(cfg)]
    logger.info("training session %s: runs %s", session.session_id, run_ids)
    return run_ids
