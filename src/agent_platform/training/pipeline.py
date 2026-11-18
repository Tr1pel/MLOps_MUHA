"""Entry point: python -m agent_platform.training.pipeline [--stage ...]"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from threadpoolctl import threadpool_limits

import mlflow
from agent_platform.training import dataset as ds
from agent_platform.training import flywire, tracking
from agent_platform.training.config import PipelineConfig
from agent_platform.training.eda import run_eda
from agent_platform.training.registry import (
    format_registry,
    promote_champion,
    register_runs,
    registry_rows,
)
from agent_platform.training.train import train_all

logger = logging.getLogger("agent_platform.training")

STAGES = ("flywire", "generate", "eda", "train", "register")


def stage_flywire(cfg: PipelineConfig, rebuild: bool) -> None:
    edges_path, neurons_path, meta_path = flywire.subgraph_paths(cfg.data_dir)
    if rebuild or not (edges_path.exists() and neurons_path.exists() and meta_path.exists()):
        flywire.build_subgraph(cfg.data_dir)
        return
    meta = flywire.load_metadata(cfg.data_dir)
    for name, expected in meta["files"].items():
        actual = flywire.sha256_of(edges_path.parent / name)
        if actual != expected:
            raise RuntimeError(f"{name} does not match metadata.json (sha256 {actual}); rebuild it")
    logger.info(
        "FlyWire subgraph verified: %d neurons, %d edges, raw sha256 %s",
        meta["counts"]["neurons"],
        meta["counts"]["edges"],
        meta["raw_file_sha256"][:12],
    )


def stage_generate(cfg: PipelineConfig) -> None:
    params = ds.GenerationParams(
        n_episodes=cfg.n_episodes, n_eval_episodes=cfg.n_eval_episodes, max_steps=cfg.max_steps
    )
    metadata = ds.generate(cfg.work_dir, params)
    tracking.log_generation_run(cfg.work_dir, cfg.data_experiment, metadata)


def stage_register(cfg: PipelineConfig) -> None:
    candidates = register_runs(cfg)
    promote_champion(cfg, candidates)
    logger.info("\n%s", format_registry(cfg.model_name, registry_rows(cfg.model_name)))


def run(cfg: PipelineConfig, stages: Sequence[str], rebuild_flywire: bool = False) -> None:
    logger.info("tracking uri: %s", mlflow.get_tracking_uri())
    for experiment in (cfg.data_experiment, cfg.experiment):
        tracking.ensure_experiment(experiment)
    with threadpool_limits(cfg.threads):
        for stage in stages:
            logger.info("=== stage: %s ===", stage)
            if stage == "flywire":
                stage_flywire(cfg, rebuild_flywire)
            elif stage == "generate":
                stage_generate(cfg)
            elif stage == "eda":
                run_eda(cfg)
            elif stage == "train":
                train_all(cfg)
            else:
                stage_register(cfg)


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(
        description="FlyWire behaviour-cloning pipeline; the MLflow server comes from "
        "MLFLOW_TRACKING_URI."
    )
    parser.add_argument("--stage", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument(
        "--work-dir", type=Path, help="generated data (default: <data-dir>/generated)"
    )
    parser.add_argument(
        "--rebuild-flywire",
        action="store_true",
        help="re-download the pinned raw table and reselect the subgraph",
    )
    parser.add_argument("--episodes", type=int, default=defaults.n_episodes)
    parser.add_argument("--eval-episodes", type=int, default=defaults.n_eval_episodes)
    parser.add_argument("--n-steps", type=int, nargs="+", default=list(defaults.n_steps))
    parser.add_argument("--experiment", default=defaults.experiment)
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--alias", default=defaults.alias)
    parser.add_argument("--threads", type=int, default=defaults.threads)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    # MLflow prints an emoji link to the run; legacy Windows code pages cannot encode it.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = PipelineConfig(
        data_dir=args.data_dir,
        work_dir=args.work_dir,
        n_episodes=args.episodes,
        n_eval_episodes=args.eval_episodes,
        n_steps=tuple(args.n_steps),
        experiment=args.experiment,
        model_name=args.model_name,
        alias=args.alias,
        threads=args.threads,
    )
    run(cfg, args.stage, args.rebuild_flywire)


if __name__ == "__main__":
    main()
