"""MNIST + PyTorch Lightning + Weights & Biases, runnable on a laptop or on Delta.

Single GPU / CPU:   python multi_node_delta.py --epochs 3
Delta (Slurm):      sbatch multi_node_job.sh
"""
import argparse
import os

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class SimpleNN(pl.LightningModule):
    def __init__(self, lr=1e-3, hidden=128):
        super().__init__()
        # Everything passed here is logged to W&B as the run's config
        self.save_hyperparameters()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        loss = F.cross_entropy(self(data), target)
        # self.log sends the value to W&B; sync_dist averages it across GPUs
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = F.cross_entropy(logits, target)
        acc = (logits.argmax(dim=1) == target).float().mean()
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_accuracy", acc, sync_dist=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


class MNISTData(pl.LightningDataModule):
    def __init__(self, batch_size=64, data_dir="./mnist_data"):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def prepare_data(self):
        # Runs on one process per node, so processes don't download at once
        datasets.MNIST(self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        self.train_set, self.val_set = random_split(
            full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--devices", type=int, default=max(torch.cuda.device_count(), 1))
    p.add_argument("--num-nodes", type=int, default=int(os.environ.get("SLURM_JOB_NUM_NODES", 1)))
    p.add_argument("--project", default="mnist-mlops")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    pl.seed_everything(42)

    # 1. Create the W&B logger. Lightning only starts a run on rank 0.
    wandb_logger = WandbLogger(
        project=args.project,
        name=args.run_name,
        save_dir="./wandb_logs",
    )
    wandb_logger.log_hyperparams(
        {"batch_size": args.batch_size, "devices": args.devices, "num_nodes": args.num_nodes}
    )

    model = SimpleNN(lr=args.lr, hidden=args.hidden)
    data = MNISTData(batch_size=args.batch_size)

    # 2. Track gradients and weights (histograms in the run's page)
    wandb_logger.watch(model, log="all", log_freq=100)

    use_gpu = torch.cuda.is_available()
    total = args.devices * args.num_nodes
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=args.devices if use_gpu else 1,
        num_nodes=args.num_nodes,
        strategy="ddp" if use_gpu and total > 1 else "auto",
        logger=wandb_logger,  # 3. Hand the logger to the Trainer
        log_every_n_steps=50,
        default_root_dir="./mnist_multi_node_demo",
    )
    trainer.fit(model, datamodule=data)

    # 4. Log a few validation predictions as images (rank 0 only)
    if trainer.is_global_zero:
        data.setup()
        images, labels = next(iter(data.val_dataloader()))
        model.eval()
        with torch.no_grad():
            preds = model.cpu()(images[:16]).argmax(dim=1)
        wandb_logger.log_image(
            key="val_predictions",
            images=[img for img in images[:16]],
            caption=[f"true={t.item()} pred={q.item()}" for t, q in zip(labels[:16], preds)],
        )


if __name__ == "__main__":
    main()
