"""The compared model variants. Every variant is a plain sklearn Pipeline ending in a classifier."""

from dataclasses import dataclass
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agent_platform.ml.features import ConnectomeFeatures
from agent_platform.training.config import PipelineConfig


@dataclass(frozen=True)
class VariantSpec:
    name: str
    family: str  # "linear", "gbdt" or "connectome"
    n_steps: int | None = None


def variant_specs(cfg: PipelineConfig) -> list[VariantSpec]:
    """baseline, a nonlinear control, and the FlyWire models with different propagation depth."""
    specs = [VariantSpec("baseline", "linear"), VariantSpec("baseline-gbdt", "gbdt")]
    specs += [VariantSpec(f"flywire-k{k}", "connectome", k) for k in cfg.n_steps]
    return specs


def build_pipeline(spec: VariantSpec, cfg: PipelineConfig, graph_path: Path) -> Pipeline:
    def logreg() -> LogisticRegression:
        return LogisticRegression(C=cfg.logreg_c, max_iter=2000, random_state=cfg.model_seed)

    if spec.family == "linear":
        return Pipeline([("scale", StandardScaler()), ("clf", logreg())])
    if spec.family == "gbdt":
        clf = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.1, random_state=cfg.model_seed
        )
        return Pipeline([("clf", clf)])
    connectome = ConnectomeFeatures(
        graph_path=str(graph_path),
        n_steps=spec.n_steps,
        n_input=cfg.n_input,
        seed=cfg.model_seed,
    )
    return Pipeline([("connectome", connectome), ("scale", StandardScaler()), ("clf", logreg())])


def describe_params(spec: VariantSpec, pipeline: Pipeline) -> dict[str, object]:
    """Flat, human-readable hyperparameters for MLflow (no file paths)."""
    params: dict[str, object] = {"model_family": spec.family, "variant": spec.name}
    clf = pipeline.named_steps["clf"]
    params["classifier"] = type(clf).__name__
    for key in ("C", "max_iter", "max_depth", "learning_rate", "random_state"):
        if hasattr(clf, key):
            params[f"clf.{key}"] = getattr(clf, key)
    if "connectome" in pipeline.named_steps:
        conn = pipeline.named_steps["connectome"]
        for key in ("n_steps", "n_input", "input_scale", "spectral_radius", "seed"):
            params[f"connectome.{key}"] = getattr(conn, key)
    return params
