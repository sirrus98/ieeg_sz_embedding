"""
Example: How to use the HDF5 dataset for training.

This demonstrates:
1. Loading the HDF5 dataset
2. Creating train/val/test splits
3. Using DataLoader for batch iteration
4. Memory-efficient training loop
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset_all_windows import HDF5WindowDataset, create_dataloader
from config_benchmark import BenchmarkConfig


def create_patient_stratified_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Create train/val/test splits stratified by patient.
    
    This ensures that all windows from the same patient are in the same split,
    preventing data leakage.
    
    Args:
        dataset: HDF5WindowDataset instance
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Get patient IDs for all samples
    print("Loading patient IDs for stratification...")
    patient_ids = dataset.get_patient_ids()
    
    # Get unique patients
    unique_patients = sorted(set(patient_ids))
    print(f"Total patients: {len(unique_patients)}")
    
    # Split patients
    train_patients, temp_patients = train_test_split(
        unique_patients, 
        test_size=(val_ratio + test_ratio),
        random_state=random_seed
    )
    
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_seed
    )
    
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")
    
    # Create indices for each split
    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
    test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_patients]
    
    print(f"\nTrain samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def example_training_loop():
    """Example of using the HDF5 dataset in a training loop."""
    
    print("="*60)
    print("Example: Using HDF5 Dataset for Training (Per-Patient Files)")
    print("="*60)
    
    config = BenchmarkConfig()
    h5_dir = os.path.join(config.OUTPUT_DATA_DIR, "all_windows_per_patient")
    
    # Check if directory exists
    if not os.path.exists(h5_dir):
        print(f"\n✗ HDF5 directory not found: {h5_dir}")
        print("  Run extract_all_windows.py first to create the dataset.")
        return
    
    print("\n1. Loading dataset...")
    dataset = HDF5WindowDataset(h5_dir, return_metadata=False)
    
    print("\n2. Creating patient-stratified splits...")
    train_dataset, val_dataset, test_dataset = create_patient_stratified_splits(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    print("\n3. Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    print("\n4. Testing batch iteration (3 batches)...")
    for i, batch in enumerate(train_loader):
        signals, coords, regs, labels = batch
        
        print(f"\n  Batch {i+1}:")
        print(f"    Signals: {signals.shape} [{signals.dtype}]")
        print(f"    Coords: {coords.shape} [{coords.dtype}]")
        print(f"    Regs: {regs.shape} [{regs.dtype}]")
        print(f"    Labels: {labels.shape} [{labels.dtype}]")
        print(f"    Label distribution: {torch.bincount(labels).tolist()}")
        print(f"    Memory usage: ~{signals.element_size() * signals.nelement() / 1024**2:.1f} MB")
        
        if i >= 2:  # Only show 3 batches
            break
    
    print("\n5. Demonstrating memory efficiency...")
    print("   Note: With per-patient HDF5 files:")
    print("   - Each patient's data saved separately")
    print("   - Only current batch loaded into memory")
    print("   - No large monolithic file")
    print(f"   Full dataset size: {len(dataset)} windows")
    print(f"   Memory per batch (32 windows): ~{32 * 101 * 2560 * 4 / 1024**2:.1f} MB")
    print(f"   Total if loaded at once: ~{len(dataset) * 101 * 2560 * 4 / 1024**3:.1f} GB")
    
    print("\n✓ Example complete!")
    print("\nTo use in your training:")
    print("  1. Import: from dataset_all_windows import HDF5WindowDataset, create_dataloader")
    print("  2. Create dataset: dataset = HDF5WindowDataset('data/all_windows_per_patient')")
    print("  3. Create splits: train/val/test = create_patient_stratified_splits(dataset)")
    print("  4. Create loaders: DataLoader(train_dataset, batch_size=32, ...)")
    print("  5. Train: for batch in train_loader: ...")


if __name__ == "__main__":
    example_training_loop()
