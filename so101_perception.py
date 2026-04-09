import logging
from functools import cached_property

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.types import RobotObservation

from perception import detect_object

logger = logging.getLogger(__name__)

# Cameras to run detection on. Must match keys in your camera config.
DETECTION_CAMERAS = ["wrist", "side", "top"]

# Colors to detect per camera.
DETECTION_COLORS = ["red", "pink"]

# Sentinel value when object is not detected
NO_DETECTION = -1.0


class SO101FollowerWithPerception(SOFollower):
    """SO101Follower that adds 2D object pose from color detection to observations.

    Runs object detection for each color on all three cameras. Adds float features
    named {cam}_{color}_u and {cam}_{color}_v for each combination, e.g.:
      - wrist_red_u, wrist_red_v, wrist_pink_u, wrist_pink_v
      - side_red_u, side_red_v, side_pink_u, side_pink_v
      - top_red_u, top_red_v, top_pink_u, top_pink_v

    These automatically merge into observation.state via hw_to_dataset_features,
    expanding it from shape (6,) to (18,) with no changes to LeRobot internals.
    """

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        base = {**self._motors_ft, **self._cameras_ft}
        for cam in DETECTION_CAMERAS:
            for color in DETECTION_COLORS:
                base[f"{cam}_{color}_u"] = float
                base[f"{cam}_{color}_v"] = float
        return base

    def get_observation(self) -> RobotObservation:
        obs = super().get_observation()

        for cam in DETECTION_CAMERAS:
            image = obs.get(cam)

            for color in DETECTION_COLORS:
                u_key = f"{cam}_{color}_u"
                v_key = f"{cam}_{color}_v"

                if image is not None:
                    result = detect_object(image, color)
                    if result is not None:
                        obs[u_key] = result["centroid"][0]
                        obs[v_key] = result["centroid"][1]
                        logger.debug(f"{cam}/{color}: object at ({result['centroid'][0]:.3f}, {result['centroid'][1]:.3f})")
                    else:
                        obs[u_key] = NO_DETECTION
                        obs[v_key] = NO_DETECTION
                        logger.debug(f"{cam}/{color}: not detected")
                else:
                    obs[u_key] = NO_DETECTION
                    obs[v_key] = NO_DETECTION
                    logger.warning(f"Camera '{cam}' not found in observation")

        return obs
