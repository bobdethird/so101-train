import time

import rerun as rr

from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.processor import make_default_processors
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

USE_CAMERA = True
DISPLAY_DATA = True
FPS = 60

camera_config = {
    "front": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30)
} if USE_CAMERA else {}

robot_config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AE60557941",
    id="my_awesome_follower_arm",
    **({"cameras": camera_config} if USE_CAMERA else {}),
)

teleop_config = SO101LeaderConfig(
    port="/dev/tty.usbmodem5AE60534461",
    id="my_awesome_leader_arm",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

robot.connect()
teleop_device.connect()

if DISPLAY_DATA:
    init_rerun(session_name="teleoperation")

try:
    while True:
        loop_start = time.perf_counter()

        obs = robot.get_observation()
        raw_action = teleop_device.get_action()
        teleop_action = teleop_action_processor((raw_action, obs))
        robot_action = robot_action_processor((teleop_action, obs))
        robot.send_action(robot_action)

        if DISPLAY_DATA:
            obs_transition = robot_observation_processor(obs)
            log_rerun_data(observation=obs_transition, action=teleop_action)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / FPS - dt_s, 0.0))
except KeyboardInterrupt:
    pass
finally:
    if DISPLAY_DATA:
        rr.rerun_shutdown()
    teleop_device.disconnect()
    robot.disconnect()
