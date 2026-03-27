"""
Training script for ictal vs interictal classification (binary).

Fine-tunes MultivarWav2Vec2 for supervised binary classification (seizure detection).
"""

import sys
import os

# Add benchmark directory to path
benchmark_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, benchmark_dir)

# Add code directory to path for model imports
code_dir = os.path.join(os.path.dirname(benchmark_dir), 'code')
if os.path.exists(code_dir):
    sys.path.insert(0, code_dir)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix, AUROC
import warnings
from pathlib import Path

from config_benchmark import BenchmarkConfig
from dataset_all_windows import NPZWindowDataset
from model_classifier import MultivarWav2Vec2Classifier

warnings.filterwarnings("ignore")


class IctalInterictalClassifierModule(L.LightningModule):
    """Lightning module for ictal vs interictal classification."""
    
    def __init__(self, config, class_weights=None, freeze_encoder=False):
        """
        Initialize the module.
        
        Args:
            config: Configuration object
            class_weights: Class weights for handling imbalance
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()
        
        self.config = config
        self.num_classes = 2
        self.save_hyperparameters(ignore=['class_weights'])
        
        # Create model
        self.model = MultivarWav2Vec2Classifier(
            num_classes=2,
            freeze_encoder=freeze_encoder
        )
        
        # Load pretrained weights if specified
        if config.PRETRAINED_CHECKPOINT is not None:
            print(f"Loading pretrained encoder from: {config.PRETRAINED_CHECKPOINT}")
            self.model.load_pretrained_encoder(config.PRETRAINED_CHECKPOINT)
        
        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        
        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.test_f1 = F1Score(task="binary")
        
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_auroc = AUROC(task="binary")
        
        self.val_confusion = ConfusionMatrix(task="binary")
        self.test_confusion = ConfusionMatrix(task="binary")
    
    def forward(self, x, regs):
        """Forward pass."""
        return self.model(x, regs)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        signals, coords, regs, labels = batch
        
        # Handle NaNs
        signals = torch.nan_to_num(signals, posinf=0.0, neginf=0.0)
        
        # Forward pass
        logits = self(signals, regs)
        
        # Loss
        loss = self.criterion(logits, labels)
        
        # Metrics (use class 1 probability for binary metrics)
        probs = F.softmax(logits, dim=1)[:, 1]
        self.train_acc(probs, labels)
        self.train_f1(probs, labels)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        signals, coords, regs, labels = batch
        
        # Handle NaNs
        signals = torch.nan_to_num(signals, posinf=0.0, neginf=0.0)
        
        # Forward pass
        logits = self(signals, regs)
        
        # Loss
        loss = self.criterion(logits, labels)
        
        # Metrics
        probs = F.softmax(logits, dim=1)[:, 1]
        self.val_acc(probs, labels)
        self.val_f1(probs, labels)
        self.val_precision(probs, labels)
        self.val_recall(probs, labels)
        self.val_confusion(probs, labels)
        
        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        signals, coords, regs, labels = batch
        
        # Handle NaNs
        signals = torch.nan_to_num(signals, posinf=0.0, neginf=0.0)
        
        # Forward pass
        logits = self(signals, regs)
        
        # Loss
        loss = self.criterion(logits, labels)
        
        # Metrics
        probs = F.softmax(logits, dim=1)[:, 1]
        self.test_acc(probs, labels)
        self.test_f1(probs, labels)
        self.test_precision(probs, labels)
        self.test_recall(probs, labels)
        self.test_auroc(probs, labels)
        self.test_confusion(probs, labels)
        
        # Logging
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.LR_FACTOR,
            patience=self.config.LR_PATIENCE
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def collate_fn(batch):
    """
    Collate function that handles variable channel counts with padding.
    
    Pads all samples to 150 channels (channel_buffer_size) to match model expectations.
    
    Args:
        batch: List of (signals, coords, regs, labels) tuples
        
    Returns:
        Tuple of batched tensors (signals_padded, coords_padded, regs_padded, labels_tensor)
    """
    signals, coords, regs, labels = zip(*batch)
    
    # Pad to fixed channel_buffer_size (150) to match model architecture
    # The model's temporal_transformer_hidden = channel_buffer_size * ft_enc_dims[-1]
    channel_buffer_size = 150
    n_timesteps = signals[0].shape[1]
    batch_size = len(signals)
    
    # Pad to channel_buffer_size
    signals_padded = torch.zeros(batch_size, channel_buffer_size, n_timesteps)
    coords_padded = torch.zeros(batch_size, channel_buffer_size, 3)
    regs_padded = torch.zeros(batch_size, channel_buffer_size, dtype=torch.long)
    
    for i, (sig, coord, reg) in enumerate(zip(signals, coords, regs)):
        n_ch = min(sig.shape[0], channel_buffer_size)  # Cap at channel_buffer_size
        signals_padded[i, :n_ch] = sig[:n_ch]
        coords_padded[i, :n_ch] = coord[:n_ch]
        regs_padded[i, :n_ch] = reg[:n_ch]
    
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return signals_padded, coords_padded, regs_padded, labels_tensor


def test_write_permissions(directory):
    """
    Test write permissions by creating a temporary test file.
    
    Args:
        directory: Directory path where we want to save data
        
    Raises:
        PermissionError: If we don't have write permissions
    """
    test_file = os.path.join(directory, '.permission_test')
    try:
        with open(test_file, 'w') as f:
            f.write("permission test")
        os.remove(test_file)
    except PermissionError as e:
        print(f"\n✗ ERROR: No write permission for {directory}")
        print(f"  {str(e)}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: Could not test write permissions for {directory}: {str(e)}")
        raise


def main(args):
    """Main training function."""
    print("=" * 60)
    print("Ictal vs Interictal Classification Training")
    print("=" * 60)
    
    # Configuration
    config = BenchmarkConfig()
    config.create_directories()
    
    # Test write permissions for checkpoint and log directories
    print("Testing write permissions...")
    test_write_permissions(config.BENCHMARK_CKPT_DIR)
    test_write_permissions(config.LOGS_DIR)
    print("✓ Write permissions confirmed!\n")
    
    # Override config with command line args
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.pretrained_ckpt is not None:
        config.PRETRAINED_CHECKPOINT = args.pretrained_ckpt
    
    # Override freeze_encoder setting (default is False for fine-tuning)
    config.FREEZE_ENCODER = args.freeze_encoder
    
    # Set random seed
    L.seed_everything(config.RANDOM_SEED)
    torch.set_float32_matmul_precision('medium')
    
    # Allow non-deterministic operations (adaptive pooling doesn't have deterministic CUDA impl)
    # Reproducibility is still maintained through random seed
    torch.use_deterministic_algorithms(False)
    
    # Load NPZ dataset
    print("\nLoading dataset...")
    data_dir_name = args.data_dir if hasattr(args, 'data_dir') and args.data_dir else "all_windows_per_patient_perpatient"
    npz_dir = Path(config.OUTPUT_DATA_DIR) / data_dir_name
    
    if not npz_dir.exists():
        raise FileNotFoundError(
            f"NPZ directory not found: {npz_dir}\n"
            f"Please run extract_all_windows.py first to generate the data.\n"
            f"For global normalization: python extract_all_windows.py --use-global-norm"
        )
    
    print(f"Loading data from: {npz_dir}")
    use_mmap = not args.no_mmap
    dataset = NPZWindowDataset(str(npz_dir), return_metadata=False, use_mmap=use_mmap)
    print(f"✓ Loaded dataset with {len(dataset)} windows")
    
    # Train/val/test split (70/15/15)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    print(f"✓ Split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=config.NUM_WORKERS > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=config.NUM_WORKERS > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=config.NUM_WORKERS > 0
    )
    
    # Get class weights from dataset (already computed during initialization)
    print("\nCalculating class weights...")
    class_weights = dataset.get_class_weights()
    class_weights = class_weights.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Class weights: {class_weights}")
    
    # Create model
    print("\nCreating model...")
    print(f"  Freeze encoder: {config.FREEZE_ENCODER}")
    print(f"  Training mode: {'Feature extraction (frozen encoder)' if config.FREEZE_ENCODER else 'Fine-tuning (entire model trainable)'}")
    
    # Load from checkpoint if specified
    if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
        print(f"  Loading from checkpoint: {args.pretrained_ckpt}")
        model = IctalInterictalClassifierModule.load_from_checkpoint(
            args.pretrained_ckpt,
            config=config,
            class_weights=class_weights,
            freeze_encoder=config.FREEZE_ENCODER,
            strict=False  # Allow partial loading
        )
        print("  ✓ Checkpoint loaded successfully")
    else:
        if args.pretrained_ckpt:
            print(f"  Warning: Checkpoint not found: {args.pretrained_ckpt}")
            print("  Creating new model instead...")
        model = IctalInterictalClassifierModule(
            config=config,
            class_weights=class_weights,
            freeze_encoder=config.FREEZE_ENCODER
        )
    
    # Callbacks
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        monitor='val_loss',
        filename='ictal_interictal-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}',
        save_top_k=3,
        mode='min',
        dirpath=os.path.join(config.BENCHMARK_CKPT_DIR, 'ictal_interictal')
    )
    
    early_stopping = pl_callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='min',
        verbose=True
    )
    
    # Loggers
    csv_logger = pl_loggers.CSVLogger(
        config.LOGS_DIR,
        name='ictal_interictal'
    )
    
    loggers = [csv_logger]
    
    # Configure W&B logger based on mode
    if args.wandb_mode != 'disabled':
        wandb_logger = pl_loggers.WandbLogger(
            project='ieeg-seizure-detection',
            name='ictal_interictal',
            save_dir='wandb_logs',
            log_model=True,  # Log model checkpoints to W&B
            mode=args.wandb_mode,  # 'online', 'offline', or 'disabled'
            config={
                'model': 'MultivarWav2Vec2Classifier',
                'data_dir': data_dir_name,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'weight_decay': config.WEIGHT_DECAY,
                'max_epochs': config.MAX_EPOCHS,
                'freeze_encoder': config.FREEZE_ENCODER,
                'precision': args.precision,
                'use_mmap': not args.no_mmap
            }
        )
        loggers.append(wandb_logger)
        print(f"✓ Configured loggers: CSV, W&B ({args.wandb_mode})")
    else:
        print("✓ Configured loggers: CSV (W&B disabled)")
    
    # Trainer
    print("\nStarting training...")
    print(f"  Precision: {args.precision}")
    trainer = L.Trainer(
        max_epochs=config.MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping],
        logger=loggers,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        deterministic=False,  # Adaptive pooling doesn't have deterministic CUDA implementation
        precision=args.precision
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print("\nTesting best model...")
    trainer.test(model, test_loader, ckpt_path='best')
    
    # Print confusion matrix
    confusion = model.test_confusion.compute()
    print("\nTest Confusion Matrix:")
    print("                 Predicted")
    print("              Interictal  Ictal")
    print(f"Actual Interictal  {confusion[0,0]:4d}    {confusion[0,1]:4d}")
    print(f"       Ictal       {confusion[1,0]:4d}    {confusion[1,1]:4d}")
    
    print("\n✓ Training complete!")
    print(f"  Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"  Test Accuracy: {model.test_acc.compute():.4f}")
    print(f"  Test F1 Score: {model.test_f1.compute():.4f}")
    print(f"  Test AUROC: {model.test_auroc.compute():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ictal/interictal classifier")
    
    parser.add_argument("--data-dir", type=str, default="all_windows_per_patient_global_norm",
                       help="Data directory name under OUTPUT_DATA_DIR (default: all_windows_per_patient, use all_windows_per_patient_global_norm for global normalization)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--pretrained_ckpt", type=str, 
                    #    default=r"C:\Users\sirrus\Desktop\ieeg_sz_embedding\checkpoints\checkpoints_10s_128_batch\model_epoch-epoch=19-train_loss=5.56027.ckpt",
                       default=None,
                       help="Path to pretrained checkpoint")
    parser.add_argument("--freeze_encoder", action="store_true", default=False,
                       help="Freeze encoder for feature extraction mode (default: False, fine-tune entire model)")
    parser.add_argument("--precision", type=str, default="16-mixed",
                       help="Training precision: 32, 16-mixed, bf16-mixed (default: 16-mixed)")
    parser.add_argument("--no_mmap", action="store_true", default=False,
                       help="Load all data into RAM instead of memory mapping (default: False, use mmap)")
    parser.add_argument("--wandb-mode", type=str, default="online",
                       choices=['online', 'offline', 'disabled'],
                       help="W&B logging mode: online, offline, or disabled (default: disabled)")
    
    args = parser.parse_args()
    
    main(args)
