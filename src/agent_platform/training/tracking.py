"""MLflow dataset tracking helpers: sources, dataset objects, generation run."""

import json
import logging
from pathlib import Path
from typing import Any

import mlflow.artifacts
import pandas as pd
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.dataset_source_registry import register_dataset_source
from mlflow.data.pandas_dataset import PandasDataset

import mlflow
from agent_platform.sim.env import ENV_VERSION
from agent_platform.training import dataset as ds
from agent_platform.training import flywire

logger = logging.getLogger(__name__)

GENERATION_RUN_FILE = "generation_run.json"
DATASET_ARTIFACT_DIR = "datasets"


class RunArtifactDatasetSource(DatasetSource):
    """A file stored as an artifact of an MLflow run (``runs:/<run_id>/<path>``).

    MLflow 3.18 has no built-in source for run artifacts (the generic ``runs`` and
    ``mlflow-artifacts`` schemes are excluded from the auto-registered artifact sources), so the
    lineage "dataset -> the run that generated it" needs this small custom source type.
    """

    def __init__(self, run_id: str, path: str) -> None:
        self.run_id = run_id
        self.path = path

    @property
    def uri(self) -> str:
        return f"runs:/{self.run_id}/{self.path}"

    @staticmethod
    def _get_source_type() -> str:
        return "mlflow_run_artifact"

    def load(self) -> str:
        return mlflow.artifacts.download_artifacts(self.uri)

    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        return isinstance(raw_source, str) and raw_source.startswith("runs:/")

    @classmethod
    def _resolve(cls, raw_source: str) -> "RunArtifactDatasetSource":
        run_id, _, path = raw_source.removeprefix("runs:/").partition("/")
        return cls(run_id, path)

    def to_dict(self) -> dict[str, Any]:
        return {"uri": self.uri, "run_id": self.run_id, "path": self.path}

    @classmethod
    def from_dict(cls, source_dict: dict[Any, Any]) -> "RunArtifactDatasetSource":
        return cls(source_dict["run_id"], source_dict["path"])


register_dataset_source(RunArtifactDatasetSource)


def ensure_experiment(name: str) -> str:
    """Create-or-reuse an experiment (safe to call on every pipeline run)."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name)


def log_generation_run(work_dir: Path, experiment: str, metadata: dict) -> str:
    """Upload the generated parquet files and parameters; remember the run id next to the data."""
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="generate-dataset") as run:
        mlflow.set_tags({"stage": "generate", "env_version": ENV_VERSION})
        mlflow.log_params({f"gen.{k}": v for k, v in metadata["generation"].items()})
        mlflow.log_params({f"teacher.{k}": v for k, v in metadata["teacher"].items()})
        mlflow.log_param("env_version", ENV_VERSION)
        mlflow.log_metrics(
            {
                "rows_train": metadata["rows"]["train"],
                "rows_test": metadata["rows"]["test"],
                "episodes_train": metadata["episodes"]["train"],
                "episodes_test": metadata["episodes"]["test"],
                "teacher_success_rate": metadata["teacher_success_rate"],
            }
        )
        for name in (ds.TRAIN_FILE, ds.TEST_FILE, ds.EPISODES_FILE):
            mlflow.log_artifact(str(work_dir / name), DATASET_ARTIFACT_DIR)
        mlflow.log_artifact(str(work_dir / ds.GENERATION_FILE), DATASET_ARTIFACT_DIR)
        run_id = run.info.run_id
    (work_dir / GENERATION_RUN_FILE).write_text(
        json.dumps({"run_id": run_id, "experiment": experiment}) + "\n", encoding="utf-8"
    )
    logger.info("generation run %s", run_id)
    return run_id


def read_generation_run_id(work_dir: Path) -> str:
    path = work_dir / GENERATION_RUN_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the 'generate' stage first")
    run_id = json.loads(path.read_text(encoding="utf-8"))["run_id"]
    try:
        mlflow.tracking.MlflowClient().get_run(run_id)
    except mlflow.exceptions.MlflowException as exc:
        raise RuntimeError(
            f"generation run {run_id} not found in {mlflow.get_tracking_uri()}: "
            "re-run the 'generate' stage against this tracking server"
        ) from exc
    return run_id


def generated_dataset(frame: pd.DataFrame, split: str, run_id: str) -> PandasDataset:
    """Dataset 'fly-bc-<split>', its source is the parquet artifact of the generation run."""
    file_name = ds.TRAIN_FILE if split == "train" else ds.TEST_FILE
    source = RunArtifactDatasetSource(run_id, f"{DATASET_ARTIFACT_DIR}/{file_name}")
    return mlflow.data.from_pandas(frame, source=source, targets=ds.TARGET, name=f"fly-bc-{split}")


def flywire_dataset(edges: pd.DataFrame) -> PandasDataset:
    """Dataset 'flywire-v783-subgraph': the committed edge list, source = pinned raw table URL."""
    return mlflow.data.from_pandas(edges, source=flywire.RAW_URL, name="flywire-v783-subgraph")


def flywire_input_tags(metadata: dict) -> dict[str, str]:
    return {
        "raw_file_sha256": metadata["raw_file_sha256"],
        "repository_commit": metadata["repository_commit"],
        "flywire_version": str(metadata["flywire_version"]),
    }
