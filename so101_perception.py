import logging
from functools import cached_property

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.types import RobotObservation

from perception import detect_red_object

logger = logging.getLogger(__name__)

# Cameras to run detection on. Must match keys in your camera config.
DETECTION_CAMERAS = ["wrist", "side", "top"]

# Sentinel value when object is not detected
NO_DETECTION = -1.0


class SO101FollowerWithPerception(SOFollower):
    """SO101Follower that adds 2D object pose from color detection to observations.

    Runs red object detection on all three cameras and adds six float features:
      - wrist_object_u, wrist_object_v: from the wrist camera
      - side_object_u, side_object_v: from the side camera
      - top_object_u, top_object_v: from the top camera

    These automatically merge into observation.state via hw_to_dataset_features,
    expanding it from shape (6,) to (12,) with no changes to LeRobot internals.
    """

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        base = {**self._motors_ft, **self._cameras_ft}
        for cam in DETECTION_CAMERAS:
            base[f"{cam}_object_u"] = float
            base[f"{cam}_object_v"] = float
        return base

    def get_observation(self) -> RobotObservation:
        obs = super().get_observation()

        for cam in DETECTION_CAMERAS:
            u_key = f"{cam}_object_u"
            v_key = f"{cam}_object_v"

            image = obs.get(cam)
            if image is not None:
                result = detect_red_object(image)
                if result is not None:
                    obs[u_key] = result["centroid"][0]
                    obs[v_key] = result["centroid"][1]
                    logger.debug(f"{cam}: object at ({result['centroid'][0]:.3f}, {result['centroid'][1]:.3f})")
                else:
                    obs[u_key] = NO_DETECTION
                    obs[v_key] = NO_DETECTION
                    logger.debug(f"{cam}: object not detected")
            else:
                obs[u_key] = NO_DETECTION
                obs[v_key] = NO_DETECTION
                logger.warning(f"Camera '{cam}' not found in observation")

        return obs
