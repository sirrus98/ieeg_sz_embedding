"""
Test script for train_ictal_interictal.py

Tests dataset loading, model forward pass, and training functionality.
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from pathlib import Path
import lightning as L
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
import tempfile
import shutil

from config_benchmark import BenchmarkConfig
from dataset_all_windows import NPZWindowDataset
from model_classifier import MultivarWav2Vec2Classifier
from train_ictal_interictal import IctalInterictalClassifierModule, collate_fn


def test_dataset_loading():
    """Test NPZ dataset loads correctly."""
    print("\n" + "="*80)
    print("TEST 1: Dataset Loading")
    print("="*80)
    
    config = BenchmarkConfig()
    npz_dir = Path(config.OUTPUT_DATA_DIR) / "all_windows_per_patient"
    
    if not npz_dir.exists():
        print(f"✗ SKIP: NPZ directory not found: {npz_dir}")
        print("  Please run extract_all_windows.py first.")
        return False
    
    try:
        # Load dataset
        dataset = NPZWindowDataset(str(npz_dir), return_metadata=False)
        print(f"✓ Loaded dataset with {len(dataset)} windows")
        
        # Check a sample
        signals, coords, regs, label = dataset[0]
        print(f"✓ Sample shape:")
        print(f"  - signals: {signals.shape}")
        print(f"  - coords: {coords.shape}")
        print(f"  - regs: {regs.shape}")
        print(f"  - label: {label}")
        
        # Check splits
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"✓ Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Test collate function
        batch_size = 4
        loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
        batch = next(iter(loader))
        signals_batch, coords_batch, regs_batch, labels_batch = batch
        
        print(f"✓ Batch shapes:")
        print(f"  - signals: {signals_batch.shape}")
        print(f"  - coords: {coords_batch.shape}")
        print(f"  - regs: {regs_batch.shape}")
        print(f"  - labels: {labels_batch.shape}")
        
        assert signals_batch.shape[0] == batch_size, "Batch size mismatch"
        assert labels_batch.shape[0] == batch_size, "Labels batch size mismatch"
        
        print("✓ TEST 1 PASSED: Dataset loading works correctly")
        return True
        
    except Exception as e:
        print(f"✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """Test model forward pass with batch data."""
    print("\n" + "="*80)
    print("TEST 2: Model Forward Pass")
    print("="*80)
    
    try:
        # Create model
        model = MultivarWav2Vec2Classifier(num_classes=2, freeze_encoder=False)
        print("✓ Model created")
        
        # Create dummy batch (must use 150 channels to match model's channel_buffer_size)
        batch_size = 2
        n_channels = 150  # Must match channel_buffer_size in model
        n_timesteps = 2560
        
        signals = torch.randn(batch_size, n_channels, n_timesteps)
        regs = torch.randint(0, 41, (batch_size, n_channels))
        
        print(f"✓ Dummy batch created:")
        print(f"  - signals: {signals.shape}")
        print(f"  - regs: {regs.shape}")
        
        # Forward pass
        logits = model(signals, regs)
        print(f"✓ Forward pass successful")
        print(f"  - logits: {logits.shape}")
        
        assert logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {logits.shape}"
        
        # Test with NaN handling
        signals_with_nan = signals.clone()
        signals_with_nan[0, 0, 0] = float('nan')
        signals_clean = torch.nan_to_num(signals_with_nan, posinf=0.0, neginf=0.0)
        logits = model(signals_clean, regs)
        print(f"✓ NaN handling works")
        
        print("✓ TEST 2 PASSED: Model forward pass works correctly")
        return True
        
    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_one_epoch():
    """Test training runs for one epoch without errors."""
    print("\n" + "="*80)
    print("TEST 3: Training One Epoch (Fast Dev Run)")
    print("="*80)
    
    config = BenchmarkConfig()
    npz_dir = Path(config.OUTPUT_DATA_DIR) / "all_windows_per_patient"
    
    if not npz_dir.exists():
        print(f"✗ SKIP: NPZ directory not found: {npz_dir}")
        return False
    
    try:
        # Load small subset for fast testing
        dataset = NPZWindowDataset(str(npz_dir), return_metadata=False)
        
        # Use only 100 samples for quick test
        subset_size = min(100, len(dataset))
        indices = list(range(subset_size))
        subset = Subset(dataset, indices)
        
        train_size = int(0.7 * len(subset))
        val_size = len(subset) - train_size
        
        train_dataset, val_dataset = random_split(
            subset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"✓ Using subset: Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # Create model
        model_net = MultivarWav2Vec2Classifier(num_classes=2, freeze_encoder=False)
        lightning_model = IctalInterictalClassifierModule(
            config=config,
            class_weights=None,
            freeze_encoder=False
        )
        
        print("✓ Model created")
        
        # Create trainer with fast_dev_run
        # Set deterministic=False to avoid adaptive pooling CUDA determinism error
        import torch
        torch.use_deterministic_algorithms(False)
        
        trainer = L.Trainer(
            fast_dev_run=True,
            accelerator='cpu',
            logger=False,
            enable_checkpointing=False,
            deterministic=False
        )
        
        print("✓ Starting fast dev run...")
        trainer.fit(lightning_model, train_loader, val_loader)
        print("✓ Fast dev run completed")
        
        print("✓ TEST 3 PASSED: Training runs without errors")
        return True
        
    except Exception as e:
        print(f"✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_training_loop():
    """Test complete training with checkpointing."""
    print("\n" + "="*80)
    print("TEST 4: Full Training Loop (2 Epochs)")
    print("="*80)
    
    config = BenchmarkConfig()
    npz_dir = Path(config.OUTPUT_DATA_DIR) / "all_windows_per_patient"
    
    if not npz_dir.exists():
        print(f"✗ SKIP: NPZ directory not found: {npz_dir}")
        return False
    
    # Create temporary directories for checkpoints and logs
    temp_dir = tempfile.mkdtemp()
    temp_ckpt_dir = os.path.join(temp_dir, 'checkpoints')
    temp_log_dir = os.path.join(temp_dir, 'logs')
    os.makedirs(temp_ckpt_dir, exist_ok=True)
    os.makedirs(temp_log_dir, exist_ok=True)
    
    try:
        # Load small subset for fast testing
        dataset = NPZWindowDataset(str(npz_dir), return_metadata=False)
        
        # Use only 200 samples for testing
        subset_size = min(200, len(dataset))
        indices = list(range(subset_size))
        subset = Subset(dataset, indices)
        
        train_size = int(0.7 * len(subset))
        val_size = int(0.15 * len(subset))
        test_size = len(subset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            subset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"✓ Using subset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # Create model
        lightning_model = IctalInterictalClassifierModule(
            config=config,
            class_weights=None,
            freeze_encoder=False
        )
        
        print("✓ Model created")
        
        # Callbacks
        checkpoint_callback = pl_callbacks.ModelCheckpoint(
            monitor='val_loss',
            filename='test-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
            dirpath=temp_ckpt_dir
        )
        
        # Logger
        csv_logger = pl_loggers.CSVLogger(
            temp_log_dir,
            name='test_ictal_interictal'
        )
        
        # Create trainer
        # Set deterministic=False to avoid adaptive pooling CUDA determinism error
        torch.use_deterministic_algorithms(False)
        
        trainer = L.Trainer(
            max_epochs=2,
            callbacks=[checkpoint_callback],
            logger=csv_logger,
            accelerator='cpu',
            log_every_n_steps=1,
            enable_progress_bar=True,
            deterministic=False
        )
        
        print("✓ Starting training for 2 epochs...")
        trainer.fit(lightning_model, train_loader, val_loader)
        print("✓ Training completed")
        
        # Check checkpoint was saved
        assert checkpoint_callback.best_model_path != "", "No checkpoint saved"
        assert os.path.exists(checkpoint_callback.best_model_path), "Checkpoint file not found"
        print(f"✓ Checkpoint saved: {checkpoint_callback.best_model_path}")
        
        # Check logs were created
        log_files = list(Path(temp_log_dir).rglob("*.csv"))
        assert len(log_files) > 0, "No log files created"
        print(f"✓ Logs created: {len(log_files)} files")
        
        # Test loading checkpoint
        loaded_model = IctalInterictalClassifierModule.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            config=config,
            class_weights=None,
            freeze_encoder=False
        )
        print("✓ Checkpoint loaded successfully")
        
        # Test with test set
        print("✓ Running test set evaluation...")
        trainer.test(loaded_model, test_loader)
        print("✓ Test evaluation completed")
        
        print("✓ TEST 4 PASSED: Full training loop works correctly")
        return True
        
    except Exception as e:
        print(f"✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"✓ Cleaned up temporary directory: {temp_dir}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TEST SUITE: train_ictal_interictal.py")
    print("="*80)
    
    results = []
    
    # Run all tests
    results.append(("Dataset Loading", test_dataset_loading()))
    results.append(("Model Forward Pass", test_model_forward()))
    results.append(("Training One Epoch", test_training_one_epoch()))
    results.append(("Full Training Loop", test_full_training_loop()))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
