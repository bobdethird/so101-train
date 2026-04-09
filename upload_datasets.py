from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASETS = ["cadenli/hippo1", "cadenli/hippo2"]

for repo_id in DATASETS:
    print(f"Uploading {repo_id}...")
    ds = LeRobotDataset(repo_id)
    ds.push_to_hub()
    print(f"  Done: {repo_id}")
