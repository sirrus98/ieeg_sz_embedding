"""
Training script for benchmark models using PyTorch Lightning.

Supports five models:
- ONSET: Lightweight seizure detector with depthwise separable TCN
- WVNT: WaveNet-based detector
- Conformer: Transformer-based detector (learned positional embeddings)
- ConformerRegions: Transformer-based detector (region embeddings)
- ST: Spatial-Temporal transformer with mean pooling

Usage:
    python train_benchmarks.py --model onset --aggregation mean
    python train_benchmarks.py --model wvnt --aggregation max --wandb-mode offline
    python train_benchmarks.py --model conformer --aggregation mean --checkpoint-every-n-steps 5000
    python train_benchmarks.py --model conformer_regions --aggregation mean --batch-size 16
    python train_benchmarks.py --model st --aggregation mean --batch-size 16
"""

import os
import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

from dataset_all_windows import NPZWindowDataset, create_dataloader
from lightning_modules.ictal_classifier import IctalClassifier
from models.wrappers import ONSETChannelAggregator, WVNTChannelAggregator, ConformerChannelAggregator, ConformerRegionsChannelAggregator, STChannelAggregator
from config_benchmark import BenchmarkConfig


def train_model(
    model_name: str,
    aggregation: str = 'max',
    checkpoint_every_n_steps: int = 3000,
    batch_size: int = 32,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,  # Use 0 for Windows
    freeze_backbone: bool = True,
    use_mmap: bool = True,
    wandb_mode: str = 'online',
    data_dir: str = 'all_windows_per_patient_perpatient'
):
    """
    Train a benchmark model.
    
    Args:
        model_name: 'onset', 'wvnt', or 'conformer'
        aggregation: 'mean' or 'max' pooling across channels
        checkpoint_every_n_steps: Save checkpoint every N training steps
        batch_size: Batch size for training
        max_epochs: Maximum number of training epochs
        learning_rate: Learning rate for AdamW optimizer
        weight_decay: Weight decay for AdamW optimizer
        num_workers: Number of dataloader workers
        freeze_backbone: Whether to freeze pretrained backbone
        use_mmap: If True, use memory mapping. If False, load all data into RAM.
        wandb_mode: 'online', 'offline', or 'disabled' for W&B logging
        data_dir: Data directory name (e.g., 'all_windows_per_patient_global_norm')
    """
    print("="*80)
    print(f"Training {model_name.upper()} with {aggregation} aggregation")
    print("="*80)
    
    config = BenchmarkConfig()
    
    # Set random seed and precision
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')
    
    # Allow non-deterministic operations (adaptive pooling doesn't have deterministic CUDA impl)
    torch.use_deterministic_algorithms(False)
    
    # Create dataset
    npz_dir = Path(config.OUTPUT_DATA_DIR) / data_dir
    
    if not npz_dir.exists():
        raise FileNotFoundError(
            f"NPZ directory not found: {npz_dir}\n"
            f"Please run extract_all_windows.py first to generate the data.\n"
            f"For global normalization: python extract_all_windows.py --use-global-norm"
        )
    
    print(f"Loading data from: {npz_dir}")
    
    dataset = NPZWindowDataset(str(npz_dir), return_metadata=False, use_mmap=use_mmap)
    print(f"\n✓ Loaded dataset with {len(dataset)} windows")
    
    # Train/val/test split (70/15/15)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    print(f"✓ Split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders with custom collate function
    def collate_fn(batch):
        """Collate function that handles variable channels with padding."""
        signals, coords, regs, labels = zip(*batch)
        
        # Find max channels in this batch
        max_ch = max(s.shape[0] for s in signals)
        n_timesteps = signals[0].shape[1]
        batch_size = len(signals)
        
        # Pad to max channels
        signals_padded = torch.zeros(batch_size, max_ch, n_timesteps)
        coords_padded = torch.zeros(batch_size, max_ch, 3)
        regs_padded = torch.zeros(batch_size, max_ch, dtype=torch.long)
        
        for i, (sig, coord, reg) in enumerate(zip(signals, coords, regs)):
            n_ch = sig.shape[0]
            signals_padded[i, :n_ch] = sig
            coords_padded[i, :n_ch] = coord
            regs_padded[i, :n_ch] = reg
        
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return signals_padded, coords_padded, regs_padded, labels_tensor
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    print(f"✓ Created dataloaders (batch_size={batch_size}, num_workers={num_workers})")
    
    # Create model
    print(f"\n✓ Creating {model_name.upper()} model...")
    
    if model_name == 'onset':
        checkpoint_path = 'benchmark/models/ONSET/best_model.pth'
        model = ONSETChannelAggregator(
            checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
            aggregation=aggregation,
            freeze_backbone=freeze_backbone
        )
    elif model_name == 'wvnt':
        checkpoint_path = 'benchmark/models/WVNT/wvnt_state.pth'
        model = WVNTChannelAggregator(
            checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
            aggregation=aggregation,
            freeze_backbone=freeze_backbone
        )
    elif model_name == 'conformer':
        # Scaled down version: smaller embedding and fewer layers
        model = ConformerChannelAggregator(
            emb_size=32,  # Reduced from 40
            depth=3,      # Reduced from 6
            aggregation=aggregation,
            freeze_backbone=freeze_backbone
        )
    elif model_name == 'conformer_regions':
        # Conformer with region embeddings (instead of positional embeddings)
        model = ConformerRegionsChannelAggregator(
            emb_size=32,
            depth=3,
            num_heads=8,
            aggregation=aggregation,  # Kept for API compatibility
            freeze_backbone=freeze_backbone,
            max_channels=150,
            max_regions=42
        )
    elif model_name == 'st':
        # Spatial-Temporal model with mean pooling (like Conformer1DDualRegions)
        model = STChannelAggregator(
            spatial_embed_dim=32,
            temporal_embed_dim=32,  # Same as spatial (using mean pooling)
            depth_spatial=3,
            depth_temporal=3,
            n_spatial_tokens=20,  # Deprecated, using mean pooling
            aggregation=aggregation,  # Kept for API compatibility, ignored
            freeze_backbone=freeze_backbone,
            max_channels=150,
            n_frames=20
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: onset, wvnt, conformer, conformer_regions, st")
    
    # Get class weights from dataset (already computed during initialization)
    print("\n✓ Calculating class weights...")
    class_weights = dataset.get_class_weights()
    print(f"  Class weights: {class_weights}")
    
    # Create Lightning module
    pl_module = IctalClassifier(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        aggregation=aggregation,
        class_weights=class_weights
    )
    
    # Callbacks
    checkpoint_dir = Path('checkpoints') / f'{model_name}_{aggregation}'
    
    # Best model checkpoint based on validation AUROC
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir / 'best',
        filename='{epoch}-{val/auroc:.3f}',
        monitor='val/auroc',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    # Periodic checkpoint every N steps
    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir / 'periodic',
        filename='{step}-{epoch}',
        every_n_train_steps=checkpoint_every_n_steps,
        save_top_k=-1  # Save all periodic checkpoints
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/auroc',
        patience=10,
        mode='max',
        verbose=True
    )
    
    print(f"\n✓ Checkpoints will be saved to: {checkpoint_dir}")
    
    # Loggers (multiple for different visualization needs)
    tb_logger = TensorBoardLogger(
        'lightning_logs',
        name=f'{model_name}_{aggregation}'
    )
    
    csv_logger = CSVLogger(
        save_dir='csv_logs',
        name=f'{model_name}_{aggregation}'
    )
    
    # Configure W&B logger based on mode
    loggers = [tb_logger, csv_logger]
    
    if wandb_mode != 'disabled':
        import wandb
        wandb_logger = WandbLogger(
            project='ieeg-seizure-detection',
            name=f'{model_name}_{aggregation}',
            save_dir='wandb_logs',
            log_model=True,  # Log model checkpoints to W&B
            mode=wandb_mode,  # 'online', 'offline', or 'disabled'
            config={
                'model': model_name,
                'data_dir': data_dir,
                'aggregation': aggregation,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'max_epochs': max_epochs,
                'freeze_backbone': freeze_backbone,
                'checkpoint_every_n_steps': checkpoint_every_n_steps
            }
        )
        loggers.append(wandb_logger)
        print(f"✓ Configured loggers: TensorBoard, W&B ({wandb_mode}), CSV")
    else:
        print("✓ Configured loggers: TensorBoard, CSV (W&B disabled)")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[best_checkpoint_callback, periodic_checkpoint_callback, early_stop_callback],
        logger=loggers,
        log_every_n_steps=10,
        precision='16-mixed' if torch.cuda.is_available() else 32,  # Mixed precision on GPU
        deterministic=False  # Adaptive pooling doesn't have deterministic CUDA implementation
    )
    
    print(f"\n✓ Trainer configured:")
    print(f"  - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Precision: {'16-mixed' if torch.cuda.is_available() else '32'}")
    
    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    trainer.fit(pl_module, train_loader, val_loader)
    
    # Test
    print("\n" + "="*80)
    print("Testing best model...")
    print("="*80 + "\n")
    
    trainer.test(pl_module, test_loader, ckpt_path='best')
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best checkpoint: {best_checkpoint_callback.best_model_path}")
    print("="*80)
    
    return trainer, pl_module


def main():
    parser = argparse.ArgumentParser(
        description='Train benchmark models for seizure detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['onset', 'wvnt', 'conformer', 'conformer_regions', 'st'],
        help='Model to train'
    )
    parser.add_argument(
        '--aggregation',
        type=str,
        default='mean',
        choices=['mean', 'max'],
        help='Channel aggregation method'
    )
    
    # Data
    parser.add_argument(
        '--data-dir',
        type=str,
        default='all_windows_per_patient_perpatient',
        help='Data directory name under OUTPUT_DATA_DIR (default: all_windows_per_patient, use all_windows_per_patient_global_norm for global normalization)'
    )
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=0, help='Dataloader workers (use 0 for Windows)')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze pretrained backbone')
    parser.add_argument('--no-mmap', action='store_true', default=False, 
                       help='Load all data into RAM instead of memory mapping (default: False, use mmap)')
    
    # Checkpointing
    parser.add_argument(
        '--checkpoint-every-n-steps',
        type=int,
        default=1000,
        help='Save checkpoint every N training steps'
    )
    
    # Logging
    parser.add_argument(
        '--wandb-mode',
        type=str,
        default='online',
        choices=['online', 'offline', 'disabled'],
        help='W&B logging mode'
    )
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        model_name=args.model,
        aggregation=args.aggregation,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        freeze_backbone=args.freeze_backbone,
        use_mmap=not args.no_mmap,
        wandb_mode=args.wandb_mode,
        data_dir=args.data_dir
    )


if __name__ == '__main__':
    main()
