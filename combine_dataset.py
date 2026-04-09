import logging
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.aggregate import aggregate_datasets

logging.basicConfig(level=logging.INFO)

HF_USER = "cadenli"
SOURCE_DATASETS = [f"{HF_USER}/hippo{i}" for i in range(1, 8)]
OUTPUT_DATASET = f"{HF_USER}/hippo_combined"

print(f"Combining {len(SOURCE_DATASETS)} datasets into {OUTPUT_DATASET}:\n")

for repo_id in SOURCE_DATASETS:
    print(f"Downloading {repo_id}...")
    LeRobotDataset(repo_id)
    print(f"  Done: {repo_id}\n")

print("All datasets downloaded. Starting aggregation...\n")
aggregate_datasets(repo_ids=SOURCE_DATASETS, aggr_repo_id=OUTPUT_DATASET)
print(f"\nAggregation complete. Uploading to HuggingFace Hub...\n")

combined = LeRobotDataset(OUTPUT_DATASET)
combined.push_to_hub()
print(f"\nDone! Dataset uploaded to: https://huggingface.co/datasets/{OUTPUT_DATASET}")
