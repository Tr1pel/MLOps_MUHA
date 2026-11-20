import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EXPERIMENT = "fly-behavior-cloning"
DEFAULT_DATA_EXPERIMENT = "fly-data"
DEFAULT_MODEL_NAME = "fly-policy"
DEFAULT_ALIAS = "champion"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class PipelineConfig:
    data_dir: Path = field(default_factory=lambda: Path(_env("FLY_DATA_DIR", "data")))
    experiment: str = field(default_factory=lambda: _env("FLY_EXPERIMENT", DEFAULT_EXPERIMENT))
    data_experiment: str = DEFAULT_DATA_EXPERIMENT
    model_name: str = field(default_factory=lambda: _env("FLY_MODEL_NAME", DEFAULT_MODEL_NAME))
    alias: str = field(default_factory=lambda: _env("FLY_MODEL_ALIAS", DEFAULT_ALIAS))
    n_episodes: int = 800
    n_eval_episodes: int = 100
    # Two propagation depths for the FlyWire models; chosen from the EDA activity curves.
    n_steps: tuple[int, ...] = (2, 6)
    n_input: int = 64
    max_steps: int = 200
    model_seed: int = 0
    logreg_c: float = 1.0
    # A model is eligible for champion only if its closed-loop success rate is within this
    # margin of the teacher's success rate on the same held-out seeds.
    success_gate_margin: float = 0.10
    work_dir: Path | None = None
    # BLAS/OpenMP threads: many tiny matrix products oversubscribe a large thread pool badly.
    threads: int = 1
    eda_max_steps: int = 10
    eda_rows: int = 4000

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.work_dir = Path(self.work_dir) if self.work_dir else self.data_dir / "generated"
