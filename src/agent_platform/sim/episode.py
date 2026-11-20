from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from agent_platform.sim.env import DEFAULT_MAX_STEPS, N_ACTIONS, FlyEnv


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    success: bool
    steps: int
    collisions: int
    trajectory: list[tuple[float, float]]
    food: tuple[float, float]
    obstacles: list[tuple[float, float, float]]


def run_episode(
    policy: Callable[[np.ndarray], int], seed: int, max_steps: int = DEFAULT_MAX_STEPS
) -> EpisodeResult:
    """Run one episode synchronously; pure CPU, no I/O."""
    env = FlyEnv(seed)
    trajectory = [(env.x, env.y)]
    while env.steps < max_steps and not env.reached_food:
        action = int(policy(env.observe()))
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"policy returned invalid action id: {action}")
        env.step(action)
        trajectory.append((env.x, env.y))
    return EpisodeResult(
        seed=seed,
        success=env.reached_food,
        steps=env.steps,
        collisions=env.collisions,
        trajectory=trajectory,
        food=env.food,
        obstacles=list(env.obstacles),
    )
