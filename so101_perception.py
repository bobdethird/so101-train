import logging
from functools import cached_property

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.types import RobotObservation

logger = logging.getLogger(__name__)


class SO101FollowerWithPerception(SOFollower):
    """SO101Follower with the default 6 perception features from the base robot."""

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def get_observation(self) -> RobotObservation:
        return super().get_observation()
