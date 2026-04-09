import logging
from functools import cached_property

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.types import RobotObservation

from perception import detect_red_object

logger = logging.getLogger(__name__)

# Camera to run detection on. Must match a key in your camera config.
# Use the fixed camera (not wrist) so pixel coords are a stable reference.
DETECTION_CAMERA = "front"

# Sentinel value when object is not detected
NO_DETECTION = -1.0


class SO101FollowerWithPerception(SOFollower):
    """SO101Follower that adds 2D object pose from color detection to observations.

    Adds two float features to observation_features:
      - object_u: normalized x-coordinate of detected object [0, 1], or -1 if not found
      - object_v: normalized y-coordinate of detected object [0, 1], or -1 if not found

    These automatically merge into observation.state via hw_to_dataset_features,
    expanding it from shape (6,) to (8,) with no changes to LeRobot internals.
    """

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        base = {**self._motors_ft, **self._cameras_ft}
        base["object_u"] = float
        base["object_v"] = float
        return base

    def get_observation(self) -> RobotObservation:
        obs = super().get_observation()

        image = obs.get(DETECTION_CAMERA)
        if image is not None:
            result = detect_red_object(image)
            if result is not None:
                obs["object_u"] = result["centroid"][0]
                obs["object_v"] = result["centroid"][1]
                logger.debug(f"Object detected at ({result['centroid'][0]:.3f}, {result['centroid'][1]:.3f})")
            else:
                obs["object_u"] = NO_DETECTION
                obs["object_v"] = NO_DETECTION
                logger.debug("Object not detected")
        else:
            obs["object_u"] = NO_DETECTION
            obs["object_v"] = NO_DETECTION
            logger.warning(f"Camera '{DETECTION_CAMERA}' not found in observation")

        return obs
