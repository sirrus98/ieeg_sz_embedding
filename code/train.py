#!/usr/bin/env python
# coding: utf-8

print("========= Loading imports =========")
import warnings
import numpy as np
import argparse

import torch
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers

from model_config import get_model_config, DURATION_CONFIGS
from model_v5 import MultivarWav2Vec2
from model_wrapper import ModelV3WrapperWithQueue
from dataset import SzDataset, get_data_path
from config import CONFIG

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Self-supervised SwAV training for iEEG seizure embeddings."
)
parser.add_argument(
    "--duration",
    type=str,
    default="10s",
    choices=list(DURATION_CONFIGS.keys()),
    help="Recording duration variant (determines pkl file and crop sizes).",
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default=None,
    help="Checkpoint sub-directory under CONFIG.ckpt_dir. "
         "Defaults to 'checkpoints_<duration>'.",
)
parser.add_argument(
    "--logs_dir",
    type=str,
    default=None,
    help="CSV logger run name. Defaults to 'log_<duration>'.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--subsample",
    type=float,
    default=1.0,
    help="Fraction of channels to keep (1 = all).",
)
parser.add_argument("--shuffle", action="store_true", default=False)
parser.add_argument(
    "--max_epochs",
    type=int,
    default=500,
    help="Maximum number of training epochs.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    help="Override MODEL_CONFIG.batch_size.",
)

# W&B
parser.add_argument(
    "--wandb_mode",
    type=str,
    default="disabled",
    choices=["online", "offline", "disabled"],
    help="W&B logging mode ('disabled' skips W&B entirely).",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="ieeg-sz-embedding",
    help="W&B project name.",
)
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default=None,
    help="W&B run name. Defaults to 'swav_<duration>'.",
)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Derived defaults
# ---------------------------------------------------------------------------
ckpt_dir_name = args.ckpt_dir or f"checkpoints_{args.duration}"
log_name = args.logs_dir or f"log_{args.duration}"
ckpt_dir_path = f"{CONFIG.ckpt_dir}/{ckpt_dir_name}"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_float32_matmul_precision("medium")

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Config and data
# ---------------------------------------------------------------------------
model_config = get_model_config(args.duration)
if args.batch_size is not None:
    model_config.batch_size = args.batch_size

data_path = get_data_path(args.duration)
print(f"========= Loading data ({args.duration}) =========")
print(f"  pkl:   {data_path}")
print(f"  crops: {model_config.multi_crop_config}")

dataset = SzDataset(
    data_path=data_path,
    model_config=model_config,
    subsample=args.subsample,
    shuffle=args.shuffle,
    transform=False,
)
dataloader = DataLoader(
    dataset,
    batch_size=model_config.batch_size,
    shuffle=False,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model_nn = MultivarWav2Vec2(model_config, subsample=args.subsample)
model = ModelV3WrapperWithQueue(model_nn, model_config=model_config)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
checkpoint_callback = pl_callbacks.ModelCheckpoint(
    monitor="train_loss",
    filename="model_epoch-{epoch:02d}-{train_loss:.5f}",
    save_top_k=-1,
    every_n_epochs=20,
    dirpath=ckpt_dir_path,
)

# ---------------------------------------------------------------------------
# Loggers
# ---------------------------------------------------------------------------
csv_logger = pl_loggers.CSVLogger(CONFIG.logs_dir, name=log_name)
loggers = [csv_logger]

if args.wandb_mode != "disabled":
    from lightning.pytorch.loggers import WandbLogger
    wandb_run_name = args.wandb_run_name or f"swav_{args.duration}"
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=wandb_run_name,
        save_dir=str(CONFIG.logs_dir),
        mode=args.wandb_mode,
        config={
            "duration": args.duration,
            "multi_crop_config": model_config.multi_crop_config,
            "batch_size": model_config.batch_size,
            "learning_rate": model_config.learning_rate,
            "max_epochs": args.max_epochs,
            "seed": args.seed,
            "subsample": args.subsample,
            "prototype_dim": model_config.prototype_dim,
            "prototype_n": model_config.prototype_n,
        },
    )
    loggers.append(wandb_logger)
    print(f"W&B: project={args.wandb_project}, run={wandb_run_name}, mode={args.wandb_mode}")
else:
    print("W&B logging disabled.")

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
trainer = L.Trainer(
    log_every_n_steps=5,
    logger=loggers,
    max_epochs=args.max_epochs,
    callbacks=[checkpoint_callback],
    accelerator="gpu",
    devices=1,
)

print("========= Starting training =========")
trainer.fit(model=model, train_dataloaders=dataloader)
