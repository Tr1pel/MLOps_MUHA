import dataclasses

import numpy as np
import pytest

from agent_platform.sim.env import OBS_DIM
from agent_platform.sim.episode import EpisodeResult, run_episode
from agent_platform.sim.teacher import FORWARD, LEFT, RIGHT, Teacher, TeacherParams


def _obs(sin=0.0, cos=1.0, dist=0.5, rays=(1.0,) * 6):
    return np.array([sin, cos, dist, *rays], dtype=np.float64)


def test_teacher_goes_forward_when_aligned_and_free():
    assert Teacher()(_obs()) == FORWARD


def test_teacher_turns_toward_food():
    assert Teacher()(_obs(sin=0.8, cos=0.6)) == LEFT
    assert Teacher()(_obs(sin=-0.8, cos=0.6)) == RIGHT


def test_teacher_turns_when_food_behind():
    assert Teacher()(_obs(sin=0.0, cos=-1.0)) in (LEFT, RIGHT)


def test_teacher_avoids_obstacle_toward_open_side():
    blocked_front_left_open_right = _obs(rays=(1.0, 1.0, 0.1, 0.1, 0.3, 0.3))
    assert Teacher()(blocked_front_left_open_right) == RIGHT
    blocked_front_right_open_left = _obs(rays=(0.3, 0.3, 0.1, 0.1, 1.0, 1.0))
    assert Teacher()(blocked_front_right_open_left) == LEFT


def test_teacher_does_not_turn_into_close_obstacle_beside_it():
    # Food is to the left but the left diagonal ray is short: keep going forward.
    obs = _obs(sin=0.8, cos=0.6, rays=(1.0, 1.0, 1.0, 1.0, 0.2, 1.0))
    assert Teacher()(obs) == FORWARD


def test_teacher_params_roundtrip():
    params = TeacherParams(front_clearance=0.4)
    assert Teacher(params).params.as_dict()["front_clearance"] == 0.4


def test_teacher_is_competent_and_world_is_nontrivial():
    results = [run_episode(Teacher(), seed) for seed in range(60)]
    success = np.mean([r.success for r in results])
    assert success >= 0.85
    # Obstacles matter: a policy that ignores sensors collides far more often.
    greedy = Teacher(TeacherParams(front_clearance=-1.0, side_clearance=-1.0))
    greedy_results = [run_episode(greedy, seed) for seed in range(60)]
    assert sum(r.collisions for r in greedy_results) > sum(r.collisions for r in results)


def test_run_episode_contract():
    result = run_episode(Teacher(), seed=3)
    assert isinstance(result, EpisodeResult)
    assert result.seed == 3
    assert len(result.trajectory) == result.steps + 1
    assert all(len(point) == 2 for point in result.trajectory)
    assert len(result.food) == 2
    assert all(len(o) == 3 for o in result.obstacles)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.seed = 4


def test_run_episode_is_deterministic():
    assert run_episode(Teacher(), 11) == run_episode(Teacher(), 11)


def test_run_episode_respects_step_limit():
    result = run_episode(lambda obs: LEFT, seed=0, max_steps=15)
    assert not result.success
    assert result.steps == 15
    assert len(result.trajectory) == 16


def test_policy_gets_one_observation_and_bad_action_is_rejected():
    seen = []

    def policy(obs):
        seen.append(obs.shape)
        return 9

    with pytest.raises(ValueError):
        run_episode(policy, 0)
    assert seen == [(OBS_DIM,)]
