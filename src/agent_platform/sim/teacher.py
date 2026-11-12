"""Rule-based teacher: steer toward the food, turn away from obstacles seen by the sensors."""

from dataclasses import asdict, dataclass

import numpy as np

# Observation indices (see env.py): 0 sin(rel), 1 cos(rel), 2 distance, 3..8 sensors from
# right (-90 deg) to left (+90 deg); positive angles are to the left.
_SIN, _COS = 0, 1
_RIGHT_SENSORS = (3, 4, 5)
_LEFT_SENSORS = (6, 7, 8)
_FRONT_SENSORS = (5, 6)
_RIGHT_DIAG, _LEFT_DIAG = 4, 7

FORWARD, LEFT, RIGHT = 0, 1, 2


@dataclass(frozen=True)
class TeacherParams:
    # Sensors are normalised by SENSOR_RANGE, so 0.3 means an obstacle ~0.9 units ahead.
    front_clearance: float = 0.3
    side_clearance: float = 0.5
    # Heading error (|sin| of relative angle) below which the teacher stops turning.
    heading_tolerance: float = 0.15

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class Teacher:
    """Memoryless policy: the decision depends only on the current observation."""

    def __init__(self, params: TeacherParams | None = None) -> None:
        self.params = params or TeacherParams()

    def __call__(self, obs: np.ndarray) -> int:
        p = self.params
        left_room = float(obs[list(_LEFT_SENSORS)].sum())
        right_room = float(obs[list(_RIGHT_SENSORS)].sum())
        if min(obs[i] for i in _FRONT_SENSORS) < p.front_clearance:
            return LEFT if left_room >= right_room else RIGHT
        facing_food = obs[_COS] > 0.0 and abs(obs[_SIN]) < p.heading_tolerance
        if facing_food:
            return FORWARD
        want_left = obs[_SIN] > 0.0 or obs[_COS] <= 0.0 and left_room >= right_room
        diag = obs[_LEFT_DIAG] if want_left else obs[_RIGHT_DIAG]
        if diag < p.side_clearance:
            return FORWARD
        return LEFT if want_left else RIGHT
