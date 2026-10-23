"""Step 4: train the autoencoder, log everything to MLflow, and register the result.

    python train_model.py                  # 10 epochs
    python train_model.py --epochs 30      # train longer

Weights & Biases is optional: it is used only if WANDB_API_KEY is set in your environment.
"""
import argparse
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from mlflow import MlflowClient
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.utils import make_grid

from config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_ALIAS, MODEL_NAME, PROCESSED_DIR
from model import AutoEncoder


class HouseDataset(Dataset):
    """Loads the 128x128 grayscale PNGs written by process_images.py."""

    def __init__(self, folder):
        self.files = sorted(Path(folder).glob("*.png"))
        self.to_tensor = transforms.ToTensor()  # uint8 0-255 -> float 0-1, shape (1, H, W)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.to_tensor(Image.open(self.files[idx]).convert("L"))


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise just measures loss."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            loss = criterion(model(x), x)  # an autoencoder is graded on rebuilding its own input
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / count


def reconstruction_grid(model, batch, device):
    """Top row: original images. Bottom row: what the model rebuilt. Returns a uint8 array."""
    model.eval()
    with torch.no_grad():
        x = batch.to(device)
        y = model(x)
    grid = make_grid(torch.cat([x.cpu(), y.cpu()]), nrow=len(x), padding=2)
    return (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def train(args):
    dataset = HouseDataset(args.data)
    if len(dataset) < 2:
        sys.exit(f"Found {len(dataset)} images in '{args.data}/'. Run process_images.py first.")

    n_val = max(1, int(0.1 * len(dataset)))
    train_set, val_set = random_split(
        dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(42)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=2, pin_memory=pin)
    preview_batch = next(iter(DataLoader(val_set, batch_size=8)))  # same images every epoch

    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        wandb.init(project="house-autoencoder", config=vars(args))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Training on {device} | {len(train_set)} train / {len(val_set)} validation images")

    with mlflow.start_run() as run:
        mlflow.log_params({**vars(args), "device": str(device), "n_train": len(train_set), "n_val": len(val_set)})

        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss = run_epoch(model, val_loader, device, criterion)
            print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            grid = reconstruction_grid(model, preview_batch, device)
            mlflow.log_image(grid, artifact_file=f"reconstructions/epoch_{epoch:03d}.png")
            if use_wandb:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "reconstructions": wandb.Image(grid)}, step=epoch)

        # Save the model and register it as a new version in the Model Registry.
        model.cpu().eval()
        info = mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=np.zeros((1, 1, 128, 128), dtype=np.float32),
            # MLflow 3 defaults to a traced "pt2" export, which can't be switched to eval mode or
            # moved between devices after loading. "pickle" stores the ordinary PyTorch module
            # (it is rebuilt from model.py when loaded, so only load models you trained yourself).
            serialization_format="pickle",
        )
        # "champion" is the version the API serves. Moving the alias deploys a new model.
        MlflowClient().set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, info.registered_model_version)
        print(f"Registered '{MODEL_NAME}' version {info.registered_model_version} -> alias '@{MODEL_ALIAS}'")
        print(f"MLflow run: {run.info.run_id}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", default=PROCESSED_DIR)
    train(parser.parse_args())
