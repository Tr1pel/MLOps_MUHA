"""Model Registry: register the runs of a training session, pick and move the champion alias."""

import logging
from dataclasses import dataclass

from mlflow.entities.model_registry import ModelVersion

import mlflow
from agent_platform.training.config import PipelineConfig
from agent_platform.training.train import SESSION_TAG
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

TAG_METRICS = (
    "test_f1_macro",
    "test_accuracy",
    "episode_success_rate",
    "episode_mean_collisions",
    "episode_mean_steps",
)


@dataclass(frozen=True)
class Candidate:
    version: int
    variant: str
    run_id: str
    test_f1_macro: float
    episode_success_rate: float
    teacher_success_rate: float


def latest_session_runs(client: MlflowClient, experiment: str) -> list:
    """Finished training runs of the most recent pipeline session, in creation order."""
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise RuntimeError(f"experiment {experiment!r} not found: run the 'train' stage first")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.stage = 'train' AND attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=200,
    )
    if not runs:
        raise RuntimeError(f"no finished training runs in {experiment!r}")
    session = runs[0].data.tags[SESSION_TAG]
    return sorted(
        (r for r in runs if r.data.tags.get(SESSION_TAG) == session),
        key=lambda r: r.info.start_time,
    )


def register_runs(cfg: PipelineConfig, client: MlflowClient | None = None) -> list[Candidate]:
    """One new model version per run of the latest session, linked to its run and metrics."""
    client = client or MlflowClient()
    candidates = []
    for run in latest_session_runs(client, cfg.experiment):
        model_id = run.data.tags["model_id"]
        version = mlflow.register_model(f"models:/{model_id}", cfg.model_name)
        variant = run.data.tags["variant"]
        metrics = run.data.metrics
        for key in TAG_METRICS:
            client.set_model_version_tag(
                cfg.model_name, version.version, key, f"{metrics[key]:.4f}"
            )
        client.set_model_version_tag(cfg.model_name, version.version, "variant", variant)
        client.set_model_version_tag(
            cfg.model_name, version.version, "dataset_digest_train", _train_digest(run)
        )
        client.update_model_version(
            cfg.model_name,
            version.version,
            description=(
                f"{variant}: test macro F1 {metrics['test_f1_macro']:.4f}, closed-loop success "
                f"{metrics['episode_success_rate']:.2f} "
                f"(teacher {metrics['teacher_success_rate']:.2f}) on "
                f"{int(run.data.params['n_eval_episodes'])} held-out seeds. Source run {run.info.run_id}."
            ),
        )
        candidates.append(
            Candidate(
                version=int(version.version),
                variant=variant,
                run_id=run.info.run_id,
                test_f1_macro=metrics["test_f1_macro"],
                episode_success_rate=metrics["episode_success_rate"],
                teacher_success_rate=metrics["teacher_success_rate"],
            )
        )
        logger.info("registered %s as %s v%s", variant, cfg.model_name, version.version)
    return candidates


def _train_digest(run) -> str:
    for dataset_input in run.inputs.dataset_inputs:
        tags = {t.key: t.value for t in dataset_input.tags}
        if tags.get("mlflow.data.context") == "training":
            return f"{dataset_input.dataset.name}@{dataset_input.dataset.digest}"
    return "unknown"


def select_champion(candidates: list[Candidate], margin: float) -> Candidate:
    """Highest test macro F1 among models whose closed-loop success is within ``margin`` of the
    teacher's success rate on the same seeds (ties: higher success rate, then newer version)."""
    eligible = [c for c in candidates if c.episode_success_rate >= c.teacher_success_rate - margin]
    if not eligible:
        raise RuntimeError("no candidate passes the episode success gate; alias not changed")
    return max(eligible, key=lambda c: (c.test_f1_macro, c.episode_success_rate, c.version))


def set_alias(model_name: str, version: int, alias: str, client: MlflowClient | None = None):
    client = client or MlflowClient()
    client.get_model_version(model_name, str(version))  # fails clearly if it does not exist
    client.set_registered_model_alias(model_name, alias, str(version))
    logger.info("alias %s -> %s v%s", alias, model_name, version)


def promote_champion(cfg: PipelineConfig, candidates: list[Candidate]) -> Candidate:
    champion = select_champion(candidates, cfg.success_gate_margin)
    set_alias(cfg.model_name, champion.version, cfg.alias)
    logger.info(
        "%s v%d (%s) is %s: test_f1_macro=%.4f, success=%.2f vs teacher %.2f",
        cfg.model_name,
        champion.version,
        champion.variant,
        cfg.alias,
        champion.test_f1_macro,
        champion.episode_success_rate,
        champion.teacher_success_rate,
    )
    return champion


def registry_rows(model_name: str, client: MlflowClient | None = None) -> list[dict]:
    client = client or MlflowClient()
    versions: list[ModelVersion] = client.search_model_versions(f"name = '{model_name}'")
    aliases: dict[int, list[str]] = {}
    for alias, version in client.get_registered_model(model_name).aliases.items():
        aliases.setdefault(int(version), []).append(alias)
    rows = []
    for mv in sorted(versions, key=lambda v: int(v.version)):
        rows.append(
            {
                "version": int(mv.version),
                "aliases": sorted(aliases.get(int(mv.version), [])),
                "variant": mv.tags.get("variant", ""),
                "run_id": mv.run_id,
                "test_f1_macro": mv.tags.get("test_f1_macro", ""),
                "episode_success_rate": mv.tags.get("episode_success_rate", ""),
                "dataset_train": mv.tags.get("dataset_digest_train", ""),
            }
        )
    return rows


def format_registry(model_name: str, rows: list[dict]) -> str:
    lines = [f"registered model: {model_name} ({len(rows)} versions)"]
    header = f"{'ver':>4} {'aliases':<12} {'variant':<14} {'f1_macro':>8} {'success':>7}  run_id"
    lines.append(header)
    for r in rows:
        lines.append(
            f"{r['version']:>4} {','.join(r['aliases']) or '-':<12} {r['variant']:<14} "
            f"{r['test_f1_macro']:>8} {r['episode_success_rate']:>7}  {r['run_id']}"
        )
    return "\n".join(lines)
