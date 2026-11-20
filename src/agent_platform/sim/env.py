"""Deterministic 2D food-search world: one point agent, circular obstacles, walls.

Observation layout (OBS_DIM = 9, everything float64):
  0  sin(angle to food relative to heading)   in [-1, 1]
  1  cos(angle to food relative to heading)   in [-1, 1]
  2  distance to food / arena diagonal        in [0, 1]
  3..8  range sensors (distance to the nearest obstacle or wall along a ray at
        SENSOR_ANGLES relative to heading), divided by SENSOR_RANGE, so 1.0 = free
"""

import math

import numpy as np

ENV_VERSION = "fly-2d-v1"
ACTIONS = ("forward", "left", "right")
N_ACTIONS = 3
DEFAULT_MAX_STEPS = 200

ARENA_SIZE = 10.0
AGENT_RADIUS = 0.2
FOOD_RADIUS = 0.4
STEP_LENGTH = 0.25
TURN_ANGLE = 0.3
SENSOR_RANGE = 3.0
SENSOR_ANGLES = tuple(math.radians(d) for d in (-90.0, -45.0, -15.0, 15.0, 45.0, 90.0))
N_SENSORS = len(SENSOR_ANGLES)
OBS_DIM = 3 + N_SENSORS
_DIAGONAL = ARENA_SIZE * math.sqrt(2.0)

_FORWARD, _LEFT, _RIGHT = 0, 1, 2


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class FlyEnv:
    """The seed fixes start pose, food position and obstacles; stepping is deterministic."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        rng = np.random.default_rng(seed)
        margin = 1.0
        while True:
            start = rng.uniform(margin, ARENA_SIZE - margin, size=2)
            food = rng.uniform(margin, ARENA_SIZE - margin, size=2)
            if np.linalg.norm(start - food) >= 5.0:
                break
        self.x, self.y = float(start[0]), float(start[1])
        self.heading = float(rng.uniform(-math.pi, math.pi))
        self.food = (float(food[0]), float(food[1]))
        self.obstacles = self._place_obstacles(rng, start, food)
        self._centers = np.array([[ox, oy] for ox, oy, _ in self.obstacles]).reshape(-1, 2)
        self._radii = np.array([r for _, _, r in self.obstacles])
        self.steps = 0
        self.collisions = 0
        self.reached_food = False

    @staticmethod
    def _place_obstacles(
        rng: np.random.Generator, start: np.ndarray, food: np.ndarray
    ) -> list[tuple[float, float, float]]:
        target = int(rng.integers(4, 8))
        placed: list[tuple[float, float, float]] = []
        for _ in range(200):
            if len(placed) == target:
                break
            radius = float(rng.uniform(0.5, 1.1))
            center = rng.uniform(0.5, ARENA_SIZE - 0.5, size=2)
            if np.linalg.norm(center - start) < radius + 1.0:
                continue
            if np.linalg.norm(center - food) < radius + FOOD_RADIUS + 0.8:
                continue
            if any(
                np.hypot(center[0] - px, center[1] - py) < radius + pr + 0.5
                for px, py, pr in placed
            ):
                continue
            placed.append((float(center[0]), float(center[1]), radius))
        return placed

    def observe(self) -> np.ndarray:
        dx, dy = self.food[0] - self.x, self.food[1] - self.y
        rel = _wrap(math.atan2(dy, dx) - self.heading)
        obs = np.empty(OBS_DIM, dtype=np.float64)
        obs[0] = math.sin(rel)
        obs[1] = math.cos(rel)
        obs[2] = min(math.hypot(dx, dy) / _DIAGONAL, 1.0)
        obs[3:] = self._ranges() / SENSOR_RANGE
        return obs

    def _ranges(self) -> np.ndarray:
        angles = self.heading + np.array(SENSOR_ANGLES)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (S, 2)
        best = np.full(N_SENSORS, SENSOR_RANGE)
        origin = np.array([self.x, self.y])
        # Walls: ray-plane intersection for the four sides of the arena.
        for axis in (0, 1):
            d = dirs[:, axis]
            with np.errstate(divide="ignore", invalid="ignore"):
                t_hi = np.where(d > 1e-12, (ARENA_SIZE - origin[axis]) / d, np.inf)
                t_lo = np.where(d < -1e-12, (0.0 - origin[axis]) / d, np.inf)
            best = np.minimum(best, np.minimum(t_hi, t_lo))
        # Circles: smallest non-negative root of |o + t d - c|^2 = r^2 (d is a unit vector).
        if len(self._radii):
            oc = origin[None, :] - self._centers  # (K, 2)
            b = dirs @ oc.T  # (S, K)
            c = np.sum(oc * oc, axis=1) - self._radii**2  # (K,)
            disc = b**2 - c[None, :]
            hit = disc >= 0.0
            t = np.where(hit, -b - np.sqrt(np.where(hit, disc, 0.0)), np.inf)
            t = np.where(t >= 0.0, t, np.inf)
            best = np.minimum(best, t.min(axis=1))
        return np.clip(best, 0.0, SENSOR_RANGE)

    def step(self, action: int) -> bool:
        """Apply an action; returns True once the food is reached."""
        self.steps += 1
        if action == _LEFT:
            self.heading = _wrap(self.heading + TURN_ANGLE)
        elif action == _RIGHT:
            self.heading = _wrap(self.heading - TURN_ANGLE)
        elif action == _FORWARD:
            nx = self.x + STEP_LENGTH * math.cos(self.heading)
            ny = self.y + STEP_LENGTH * math.sin(self.heading)
            if self._blocked(nx, ny):
                self.collisions += 1
            else:
                self.x, self.y = nx, ny
        else:
            raise ValueError(f"unknown action id: {action!r}")
        if math.hypot(self.food[0] - self.x, self.food[1] - self.y) <= FOOD_RADIUS:
            self.reached_food = True
        return self.reached_food

    def _blocked(self, x: float, y: float) -> bool:
        if not (AGENT_RADIUS <= x <= ARENA_SIZE - AGENT_RADIUS):
            return True
        if not (AGENT_RADIUS <= y <= ARENA_SIZE - AGENT_RADIUS):
            return True
        if not len(self._radii):
            return False
        dist = np.hypot(self._centers[:, 0] - x, self._centers[:, 1] - y)
        return bool(np.any(dist < self._radii + AGENT_RADIUS))
