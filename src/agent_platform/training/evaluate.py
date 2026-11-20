"""Offline (per-row) and closed-loop (per-episode) evaluation of a fitted pipeline."""

import time
from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from agent_platform.sim.env import ACTIONS, N_ACTIONS
from agent_platform.sim.episode import EpisodeResult, run_episode

LABELS = list(range(N_ACTIONS))


def model_policy(model) -> Callable[[np.ndarray], int]:
    return lambda obs: int(model.predict(obs.reshape(1, -1))[0])


def offline_metrics(model, x_train, y_train, x_test, y_test) -> tuple[dict[str, float], dict]:
    """Primary metric test_f1_macro plus auxiliary row-level metrics; also raw diagnostics."""
    pred_test = model.predict(x_test)
    pred_train = model.predict(x_train)
    proba = model.predict_proba(x_test)
    metrics = {
        "test_f1_macro": f1_score(y_test, pred_test, average="macro", labels=LABELS),
        "test_accuracy": accuracy_score(y_test, pred_test),
        "test_log_loss": log_loss(y_test, proba, labels=LABELS),
        "train_f1_macro": f1_score(y_train, pred_train, average="macro", labels=LABELS),
    }
    metrics["f1_macro_gap"] = metrics["train_f1_macro"] - metrics["test_f1_macro"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_test, labels=LABELS, zero_division=0
    )
    for i, name in enumerate(ACTIONS):
        metrics[f"test_precision_{name}"] = precision[i]
        metrics[f"test_recall_{name}"] = recall[i]
        metrics[f"test_f1_{name}"] = f1[i]
    diagnostics = {
        "confusion": confusion_matrix(y_test, pred_test, labels=LABELS),
        "report": classification_report(
            y_test,
            pred_test,
            labels=LABELS,
            target_names=list(ACTIONS),
            output_dict=True,
            zero_division=0,
        ),
    }
    return {k: float(v) for k, v in metrics.items()}, diagnostics


def run_episodes(policy, seeds: range, max_steps: int) -> list[EpisodeResult]:
    return [run_episode(policy, seed, max_steps) for seed in seeds]


def episode_metrics(results: list[EpisodeResult], prefix: str = "episode") -> dict[str, float]:
    successes = [r for r in results if r.success]
    return {
        f"{prefix}_success_rate": len(successes) / len(results),
        f"{prefix}_mean_collisions": float(np.mean([r.collisions for r in results])),
        f"{prefix}_mean_steps": float(np.mean([r.steps for r in results])),
        f"{prefix}_mean_steps_successful": (
            float(np.mean([r.steps for r in successes])) if successes else float("nan")
        ),
    }


def latency_ms(model, sample: np.ndarray, repeats: int = 200) -> float:
    """Mean wall time of predict() on a single observation, in milliseconds."""
    row = sample[:1]
    model.predict(row)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(row)
    return (time.perf_counter() - start) / repeats * 1000.0
