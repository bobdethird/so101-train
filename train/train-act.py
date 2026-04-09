import modal
import subprocess

MINUTES = 60
HOURS = 60 * MINUTES

HF_USER = "cadenli"
DATASET_REPO_ID = f"{HF_USER}/hippo_combined"
POLICY_REPO_ID = f"{HF_USER}/act_hippo_combined_policy_v4"
JOB_NAME = "act_hippo_combined_v4"
OUTPUT_DIR = "/outputs/train/act_hippo_combined_v4"
LEROBOT_DIR = "/lerobot"
DATA_DIR = "/data"

app = modal.App("so101-act-training")

training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6", "libgl1")
    .pip_install("torch", "torchvision", "torchaudio")
    .run_commands(
        "git clone https://github.com/huggingface/lerobot.git /lerobot",
        "cd /lerobot && pip install -e .",
    )
    .pip_install("wandb", "huggingface_hub[cli]")
)

output_volume = modal.Volume.from_name(
    "so101-training-outputs", create_if_missing=True
)
data_volume = modal.Volume.from_name(
    "so101-dataset-cache", create_if_missing=True
)


@app.function(
    image=training_image,
    gpu="H100",
    cpu=8.0,
    volumes={
        "/outputs": output_volume,
        DATA_DIR: data_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=4 * HOURS,
)
def train(
    dataset_repo_id: str = DATASET_REPO_ID,
    policy_repo_id: str = POLICY_REPO_ID,
    batch_size: int = 64,
    steps: int = 100_000,
    wandb_enabled: bool = True,
):
    import os
    import torch
    from pathlib import Path

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from huggingface_hub import login as hf_login
    hf_login(token=os.environ["HF_TOKEN"])

    if wandb_enabled:
        import wandb
        wandb.login(key=os.environ["WANDB_API_KEY"])

    data_volume.reload()

    dataset_path = Path(DATA_DIR) / dataset_repo_id
    use_local = dataset_path.exists()
    if not use_local:
        print(f"Downloading dataset to volume...")
        subprocess.run(
            [
                "python", "-c",
                f"from lerobot.datasets.lerobot_dataset import LeRobotDataset; "
                f"LeRobotDataset('{dataset_repo_id}', root='{DATA_DIR}')",
            ],
            check=True,
            cwd=LEROBOT_DIR,
        )
        data_volume.commit()
    else:
        print(f"Using cached dataset at {dataset_path}")

    cmd = [
        "python",
        "src/lerobot/scripts/lerobot_train.py",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={DATA_DIR}",
        "--policy.type=act",
        f"--batch_size={batch_size}",
        f"--output_dir={OUTPUT_DIR}",
        f"--job_name={JOB_NAME}",
        "--policy.device=cuda",
        f"--wandb.enable={wandb_enabled}",
        f"--policy.repo_id={policy_repo_id}",
        f"--steps={steps}",
        "--num_workers=8",
    ]

    print(f"Running training:\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=LEROBOT_DIR)
    output_volume.commit()


@app.local_entrypoint()
def main(
    dataset_repo_id: str = DATASET_REPO_ID,
    policy_repo_id: str = POLICY_REPO_ID,
    batch_size: int = 64,
    steps: int = 100_000,
    wandb: bool = True,
):
    train.remote(
        dataset_repo_id=dataset_repo_id,
        policy_repo_id=policy_repo_id,
        batch_size=batch_size,
        steps=steps,
        wandb_enabled=wandb,
    )
