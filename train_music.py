# Modal wrapper for training ACE-Step LoRA models
# This wraps the trainer.py functionality to run on Modal

import modal
from typing import Optional

# Define the training image with all necessary dependencies
# We'll mount the trainer.py file using a Mount from the local directory
train_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0",
        "diffusers>=0.21.0",
        "peft>=0.6.0",
        "loguru>=0.7.0",
        "tensorboard>=2.13.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "hf_transfer>=0.1.9",
        "git+https://github.com/ace-step/ACE-Step.git@6ae0852b1388de6dc0cca26b31a86d711f723cb3",
    )
    .run_commands(
        [
            "git clone https://github.com/ace-step/ACE-Step.git /root/ACE-Step",
            "pip install -e /root/ACE-Step",
        ]
    )
)

# Define volumes for persistent storage
trained_cache_dir = "/root/.cache/ace-step/pretrained-checkpoints"
trained_cache = modal.Volume.from_name("checkpoints", create_if_missing=True)

# Volumes for datasets, logs, and configs
dataset_volume = modal.Volume.from_name("datasets", create_if_missing=True)
logs_volume = modal.Volume.from_name("training-logs", create_if_missing=True)
config_volume = modal.Volume.from_name("configs", create_if_missing=True)


app = modal.App("controlla-train-music")


@app.function(
    image=train_image,
    gpu="A100",  # Use A100 GPU for training
    volumes={
        trained_cache_dir: trained_cache,
        "/data": dataset_volume,
        "/logs": logs_volume,
        "/configs": config_volume,
    },
    timeout=86400,  # 24 hours timeout for long training runs
    allow_concurrent_inputs=1,  # Only one training job at a time
)
def train(
    # Training hyperparameters
    learning_rate: float = 1e-4,
    num_workers: int = 4,
    shift: float = 3.0,
    max_steps: int = 200000,
    every_n_train_steps: int = 2000,
    every_plot_step: int = 2000,
    epochs: int = -1,
    precision: str = "32",
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float = 0.5,
    gradient_clip_algorithm: str = "norm",
    reload_dataloaders_every_n_epochs: int = 1,
    val_check_interval: Optional[int] = None,
    # Dataset and paths
    dataset_path: str = "/data/lora_dataset",  # Path in Modal volume
    exp_name: str = "lora",
    logger_dir: str = "/logs",  # Path in Modal volume
    checkpoint_dir: Optional[str] = None,  # Will default to trained_cache_dir
    ckpt_path: Optional[str] = None,  # Path to resume from checkpoint
    lora_config_path: str = "/configs/lora_config.json",  # Path in Modal volume
    # Multi-GPU settings (for future use)
    num_nodes: int = 1,
    devices: int = 1,
):
    """
    Train a LoRA adapter for ACE-Step model on Modal.

    This function wraps the trainer.py main() function and runs it with the provided arguments.
    """
    import subprocess, sys
    import os

    # Set checkpoint_dir default if not provided
    if checkpoint_dir is None:
        checkpoint_dir = trained_cache_dir

    # Set environment variables for Hugging Face cache
    os.environ["HF_HUB_CACHE"] = trained_cache_dir
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Create a namespace object to mimic argparse.Namespace
    class Args:
        def __init__(self):
            self.num_nodes = num_nodes
            self.shift = shift
            self.learning_rate = learning_rate
            self.num_workers = num_workers
            self.epochs = epochs
            self.max_steps = max_steps
            self.every_n_train_steps = every_n_train_steps
            self.dataset_path = dataset_path
            self.exp_name = exp_name
            self.precision = precision
            self.accumulate_grad_batches = accumulate_grad_batches
            self.devices = devices
            self.logger_dir = logger_dir
            self.ckpt_path = ckpt_path
            self.checkpoint_dir = checkpoint_dir
            self.gradient_clip_val = gradient_clip_val
            self.gradient_clip_algorithm = gradient_clip_algorithm
            self.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs
            self.every_plot_step = every_plot_step
            self.val_check_interval = val_check_interval
            self.lora_config_path = lora_config_path

    args = Args()

    # Ensure directories exist
    os.makedirs(logger_dir, exist_ok=True)
    os.makedirs(os.path.dirname(lora_config_path), exist_ok=True)

    print(f"Starting training with experiment name: {exp_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Logger dir: {logger_dir}")
    print(f"LoRA config path: {lora_config_path}")

    # Commit volumes before training starts
    trained_cache.commit()
    dataset_volume.commit()
    logs_volume.commit()
    config_volume.commit()

    # Run training
    try:
        convert_cmd = [
            sys.executable,
            "/root/ACE-Step/convert2hf_dataset.py",
            "--data_dir",
            "/data/dataset",
            "--repeat_count",
            "2",
            "--output_name",
            dataset_path,
        ]
        subprocess.check_call(convert_cmd)
        dataset_volume.commit()
        dataset_volume.reload()
        base_cmd = [sys.executable, "/root/ACE-Step/trainer.py"]
        for key, value in vars(args).items():
            if value is None:
                continue

            flag = f"--{key}"
            base_cmd.append(flag)

            # booleans: include flag only if True
            if isinstance(value, bool):
                if value:
                    continue
                else:
                    base_cmd.pop()  # remove the flag
            else:
                base_cmd.append(str(value))
        print(f"Running command: {' '.join(base_cmd)}")
        subprocess.check_call(base_cmd)
        print("Training completed successfully!")

        # Commit volumes after training to save checkpoints and logs
        trained_cache.commit()
        logs_volume.commit()
        dataset_volume.commit()
        config_volume.commit()

        return {"status": "success", "exp_name": exp_name}
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()

        # Still commit volumes to save any partial progress
        trained_cache.commit()
        logs_volume.commit()

        raise


@app.local_entrypoint()
def train_main(
    # Training hyperparameters
    learning_rate: float = 1e-4,
    num_workers: int = 4,
    shift: float = 3.0,
    max_steps: int = 200000,
    every_n_train_steps: int = 2000,
    every_plot_step: int = 2000,
    epochs: int = -1,
    precision: str = "32",
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float = 0.5,
    gradient_clip_algorithm: str = "norm",
    reload_dataloaders_every_n_epochs: int = 1,
    val_check_interval: Optional[int] = None,
    # Dataset and paths
    dataset_path: str = "/data/lora_dataset",
    exp_name: str = "lora",
    logger_dir: str = "/logs",
    checkpoint_dir: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    lora_config_path: str = "/configs/lora_config.json",
    # Multi-GPU settings
    num_nodes: int = 1,
    devices: int = 1,
):
    """
    Local entrypoint to trigger training on Modal.

    Usage:
        modal run train_music.py --exp-name my_experiment --max-steps 100000
    """
    print("🚀 Starting training job on Modal...")

    result = train.remote(
        learning_rate=learning_rate,
        num_workers=num_workers,
        shift=shift,
        max_steps=max_steps,
        every_n_train_steps=every_n_train_steps,
        every_plot_step=every_plot_step,
        epochs=epochs,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
        val_check_interval=val_check_interval,
        dataset_path=dataset_path,
        exp_name=exp_name,
        logger_dir=logger_dir,
        checkpoint_dir=checkpoint_dir,
        ckpt_path=ckpt_path,
        lora_config_path=lora_config_path,
        num_nodes=num_nodes,
        devices=devices,
    )

    print(f"✅ Training completed: {result}")


# Optional: Function to upload dataset/config files to volumes
@app.function(
    image=modal.Image.debian_slim(python_version="3.10").uv_pip_install("requests"),
    volumes={
        "/data": dataset_volume,
        "/configs": config_volume,
    },
)
def upload_files(
    dataset_path: Optional[str] = None,
    config_path: Optional[str] = None,
):
    """
    Helper function to upload dataset and config files to Modal volumes.
    This should be run from your local machine to sync files.

    Note: For large datasets, consider using Modal's volume sync features
    or uploading files through other means (S3, etc.) and syncing to volumes.
    """
    if dataset_path:
        print(f"Uploading dataset from {dataset_path} to /data...")
        # check if url
        if dataset_path.startswith("http"):
            # download the file
            import requests
            from urllib.parse import unquote
            import os

            response = requests.get(dataset_path)
            response.raise_for_status()
            file_raw_name = dataset_path.split("/")[-1]
            file_name = unquote(file_raw_name)
            save_path = f"/data/dataset/{file_name}"
            print(f"Saving dataset url {dataset_path} to {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(response.content)

        else:
            import shutil

            # copy the file to the volume
            shutil.copy(dataset_path, "/data/dataset")

    if config_path:
        print(f"Uploading config from {config_path} to /configs...")
        # check if url
        if config_path.startswith("http"):
            # download the file
            import requests
            from urllib.parse import unquote
            import os

            response = requests.get(config_path)
            response.raise_for_status()
            file_raw_name = config_path.split("/")[-1]
            file_name = unquote(file_raw_name)
            save_path = f"/configs/{file_name}"
            print(f"Saving config url {config_path} to {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(response.content)

        else:
            import shutil

            # copy the file to the volume
            shutil.copy(config_path, "/configs/lora_config.json")

    dataset_volume.commit()
    config_volume.commit()
    print("Files uploaded successfully!")
