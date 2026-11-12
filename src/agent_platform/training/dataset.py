"""Behaviour-cloning dataset: teacher rollouts split by episode."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agent_platform.sim.env import ACTIONS, DEFAULT_MAX_STEPS, ENV_VERSION, FlyEnv
from agent_platform.sim.teacher import Teacher, TeacherParams

logger = logging.getLogger(__name__)

FEATURE_NAMES = (
    "food_sin",
    "food_cos",
    "food_dist",
    "ray_m90",
    "ray_m45",
    "ray_m15",
    "ray_p15",
    "ray_p45",
    "ray_p90",
)
TARGET = "action"
TRAIN_FILE = "train.parquet"
TEST_FILE = "test.parquet"
EPISODES_FILE = "episodes.parquet"
GENERATION_FILE = "generation.json"


@dataclass(frozen=True)
class GenerationParams:
    n_episodes: int = 800
    seed_start: int = 0
    test_fraction: float = 0.25
    split_seed: int = 0
    max_steps: int = DEFAULT_MAX_STEPS
    # Closed-loop evaluation seeds live far away from the dataset seeds, so no model is ever
    # evaluated on a scenario that appears in its training or test rows.
    eval_seed_start: int = 1_000_000
    n_eval_episodes: int = 100


def rollout_teacher(
    seeds: range, max_steps: int, teacher: Teacher
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-step (observation, teacher action) rows and one summary row per episode."""
    rows: list[np.ndarray] = []
    episodes = []
    for seed in seeds:
        env = FlyEnv(seed)
        n_rows = 0
        while env.steps < max_steps and not env.reached_food:
            obs = env.observe()
            action = teacher(obs)
            rows.append(np.concatenate(([seed, env.steps], obs, [action])))
            env.step(action)
            n_rows += 1
        episodes.append(
            {
                "episode_seed": seed,
                "success": env.reached_food,
                "steps": env.steps,
                "collisions": env.collisions,
                "n_obstacles": len(env.obstacles),
            }
        )
    columns = ["episode_seed", "step", *FEATURE_NAMES, TARGET]
    frame = pd.DataFrame(np.array(rows), columns=columns)
    frame[["episode_seed", "step", TARGET]] = frame[["episode_seed", "step", TARGET]].astype(
        np.int64
    )
    return frame, pd.DataFrame(episodes)


def split_by_episode(
    frame: pd.DataFrame, test_fraction: float, split_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every episode goes entirely to train or entirely to test."""
    seeds = np.sort(frame["episode_seed"].unique())
    rng = np.random.default_rng(split_seed)
    n_test = max(1, round(len(seeds) * test_fraction))
    test_seeds = set(rng.choice(seeds, size=n_test, replace=False).tolist())
    is_test = frame["episode_seed"].isin(test_seeds)
    return frame[~is_test].reset_index(drop=True), frame[is_test].reset_index(drop=True)


def generate(
    work_dir: Path, params: GenerationParams, teacher_params: TeacherParams | None = None
) -> dict:
    """Write train/test/episodes parquet and generation.json into work_dir; return the metadata."""
    teacher_params = teacher_params or TeacherParams()
    teacher = Teacher(teacher_params)
    seeds = range(params.seed_start, params.seed_start + params.n_episodes)
    frame, episodes = rollout_teacher(seeds, params.max_steps, teacher)
    train, test = split_by_episode(frame, params.test_fraction, params.split_seed)
    work_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(work_dir / TRAIN_FILE, index=False)
    test.to_parquet(work_dir / TEST_FILE, index=False)
    episodes.to_parquet(work_dir / EPISODES_FILE, index=False)
    metadata = {
        "env_version": ENV_VERSION,
        "actions": list(ACTIONS),
        "features": list(FEATURE_NAMES),
        "generation": asdict(params),
        "teacher": teacher_params.as_dict(),
        "rows": {"train": len(train), "test": len(test)},
        "episodes": {
            "train": int(train["episode_seed"].nunique()),
            "test": int(test["episode_seed"].nunique()),
        },
        "teacher_success_rate": float(episodes["success"].mean()),
    }
    (work_dir / GENERATION_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("dataset written: %s", metadata["rows"])
    return metadata


def load_split(work_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(work_dir / TRAIN_FILE), pd.read_parquet(work_dir / TEST_FILE)


def load_metadata(work_dir: Path) -> dict:
    return json.loads((work_dir / GENERATION_FILE).read_text(encoding="utf-8"))
