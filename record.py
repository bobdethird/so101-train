from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO101FollowerConfig
from so101_perception import SO101FollowerWithPerception
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

# --------------- Project setup ---------------
print("=== SO-101 Recording Setup ===\n")
HF_USER = input("HuggingFace username: ").strip()
DATASET_NAME = input("Dataset name: ").strip()
REPO_ID = f"{HF_USER}/{DATASET_NAME}"
TASK_DESCRIPTION = input("Task description: ").strip()
print(f"\nRecording to: {REPO_ID}")
print(f"Task: {TASK_DESCRIPTION}\n")

# --------------- Constants ---------------
NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 60

DISPLAY_DATA = True
PUSH_TO_HUB = True

USE_VIDEOS = True
STREAMING_ENCODING = True
ENCODER_THREADS = 2
VCODEC = "libsvtav1"
# ------------------------------------------

robot_config = SO101FollowerConfig(
    id="my_awesome_follower_arm",
    cameras={
        "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        "side": OpenCVCameraConfig(index_or_path=3, width=640, height=480, fps=30),
        "top": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),  # TODO: update index to match your top camera
    },
    port="/dev/tty.usbmodem5AE60557941",
)

teleop_config = SO101LeaderConfig(
    id="my_awesome_leader_arm",
    port="/dev/tty.usbmodem5AE60534461",
)

robot = SO101FollowerWithPerception(robot_config)
teleop = SO101Leader(teleop_config)

action_features = hw_to_dataset_features(robot.action_features, "action", USE_VIDEOS)
obs_features = hw_to_dataset_features(robot.observation_features, "observation", USE_VIDEOS)
dataset_features = {**action_features, **obs_features}

dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=USE_VIDEOS,
    image_writer_threads=4,
    streaming_encoding=STREAMING_ENCODING,
    encoder_threads=ENCODER_THREADS,
    vcodec=VCODEC,
)

_, events = init_keyboard_listener()

if DISPLAY_DATA:
    init_rerun(session_name="recording")

robot.connect()
teleop.connect()

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=DISPLAY_DATA,
    )

    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=DISPLAY_DATA,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    events["exit_early"] = False

    if dataset.has_pending_frames():
        dataset.save_episode()
        episode_idx += 1
    else:
        log_say("No frames recorded, skipping episode")
        dataset.clear_episode_buffer()

log_say("Stop recording")
try:
    robot.disconnect()
except RuntimeError as e:
    print(f"Warning: robot disconnect error (data is saved): {e}")
try:
    teleop.disconnect()
except RuntimeError as e:
    print(f"Warning: teleop disconnect error (data is saved): {e}")
dataset.finalize()
if PUSH_TO_HUB:
    dataset.push_to_hub()
