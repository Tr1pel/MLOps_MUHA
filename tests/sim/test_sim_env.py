import math

import numpy as np
import pytest

from agent_platform.sim import env as sim_env
from agent_platform.sim.env import (
    ACTIONS,
    ARENA_SIZE,
    DEFAULT_MAX_STEPS,
    ENV_VERSION,
    N_ACTIONS,
    OBS_DIM,
    FlyEnv,
)


def test_constants():
    assert ACTIONS == ("forward", "left", "right")
    assert N_ACTIONS == 3
    assert DEFAULT_MAX_STEPS == 200
    assert isinstance(ENV_VERSION, str) and ENV_VERSION
    assert OBS_DIM == 9


def test_same_seed_same_world():
    a, b = FlyEnv(7), FlyEnv(7)
    assert (a.x, a.y, a.heading, a.food, a.obstacles) == (b.x, b.y, b.heading, b.food, b.obstacles)
    np.testing.assert_array_equal(a.observe(), b.observe())


def test_different_seeds_differ():
    assert FlyEnv(1).food != FlyEnv(2).food


def test_does_not_touch_global_rng():
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    FlyEnv(3).observe()
    np.testing.assert_array_equal(before, np.random.get_state()[1])


@pytest.mark.parametrize("seed", range(20))
def test_observation_contract_and_world_validity(seed):
    env = FlyEnv(seed)
    obs = env.observe()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float64
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    assert math.isclose(obs[0] ** 2 + obs[1] ** 2, 1.0)
    assert math.dist((env.x, env.y), env.food) >= 5.0
    for ox, oy, r in env.obstacles:
        assert math.dist((env.x, env.y), (ox, oy)) > r + sim_env.AGENT_RADIUS
        assert math.dist(env.food, (ox, oy)) > r + sim_env.FOOD_RADIUS


def test_world_has_obstacles():
    counts = [len(FlyEnv(s).obstacles) for s in range(30)]
    assert min(counts) >= 3


def test_turning_changes_heading_only():
    env = FlyEnv(0)
    pos, heading = (env.x, env.y), env.heading
    env.step(1)
    assert (env.x, env.y) == pos
    assert env.heading == pytest.approx(
        (heading + sim_env.TURN_ANGLE + math.pi) % (2 * math.pi) - math.pi
    )
    env.step(2)
    assert env.heading == pytest.approx(heading, abs=1e-9)


def test_forward_moves_by_step_length():
    env = FlyEnv(0)
    # Face the arena centre so the first step cannot hit a wall.
    env.heading = math.atan2(ARENA_SIZE / 2 - env.y, ARENA_SIZE / 2 - env.x)
    env._centers = env._centers[:0]
    env._radii = env._radii[:0]
    start = (env.x, env.y)
    env.step(0)
    assert math.dist(start, (env.x, env.y)) == pytest.approx(sim_env.STEP_LENGTH)
    assert env.collisions == 0


def test_wall_blocks_and_counts_collision():
    env = FlyEnv(0)
    env.x, env.y, env.heading = sim_env.AGENT_RADIUS, 5.0, math.pi  # facing the left wall
    env.step(0)
    assert env.collisions == 1
    assert env.x == sim_env.AGENT_RADIUS


def test_obstacle_blocks_forward_move():
    env = FlyEnv(0)
    ox, oy, r = env.obstacles[0]
    env.x, env.y = ox - r - sim_env.AGENT_RADIUS - 0.05, oy
    env.heading = 0.0
    env.step(0)
    assert env.collisions == 1
    assert env.x == pytest.approx(ox - r - sim_env.AGENT_RADIUS - 0.05)


def test_range_sensors_see_obstacle_and_wall():
    env = FlyEnv(0)
    env.obstacles.clear()
    env._centers = env._centers[:0]
    env._radii = env._radii[:0]
    env.x, env.y, env.heading = 5.0, 5.0, 0.0
    obs = env.observe()
    assert np.all(obs[3:] > 0.99)  # nothing within 3 units except far walls
    env.x = ARENA_SIZE - 1.0  # wall one unit ahead
    assert env.observe()[5] == pytest.approx(
        1.0 / sim_env.SENSOR_RANGE / math.cos(math.radians(15))
    )
    env._centers = np.array([[env.x - 2.0, 5.0]])
    env._radii = np.array([0.8])
    env.heading = math.pi
    assert env.observe()[5:7].max() < 0.6  # circle 1.5 units ahead along the heading


def test_reaching_food_sets_flag():
    env = FlyEnv(0)
    env.x, env.y = env.food[0] - 0.3, env.food[1]
    env.heading = 0.0
    env._centers = env._centers[:0]
    env._radii = env._radii[:0]
    assert env.step(0) is True
    assert env.reached_food


def test_unknown_action_rejected():
    with pytest.raises(ValueError):
        FlyEnv(0).step(5)
