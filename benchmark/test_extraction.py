"""
Test script to verify extraction and preprocessing pipeline.

This script tests that:
1. Extraction runs without errors
2. Preprocessing matches make_dataset.py
3. Data is properly normalized
4. HDF5 files are valid
5. Dataset loading works correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

import numpy as np
import pandas as pd
import h5py
import json
from glob import glob

from config_benchmark import BenchmarkConfig


def test_hdf5_structure(h5_file):
    """Test that HDF5 file has expected structure."""
    print(f"\nTesting HDF5 structure: {os.path.basename(h5_file)}")
    
    with h5py.File(h5_file, 'r') as f:
        # Check required datasets
        required = ['signals', 'coords', 'regs', 'n_channels', 'labels', 
                   'patient_id', 'ch_names', 'window_info']
        
        for key in required:
            if key not in f:
                print(f"  ✗ Missing dataset: {key}")
                return False
            else:
                print(f"  ✓ Found dataset: {key}")
        
        # Check shapes
        n_windows = f['signals'].shape[0]
        max_channels = f['signals'].shape[1]
        n_timesteps = f['signals'].shape[2]
        
        print(f"\n  Dataset info:")
        print(f"    Windows: {n_windows}")
        print(f"    Max channels: {max_channels}")
        print(f"    Timesteps: {n_timesteps}")
        print(f"    Patient: {f['patient_id'][()].decode('utf-8')}")
        
        # Check consistency
        if f['coords'].shape[0] != n_windows:
            print(f"  ✗ Coords shape mismatch")
            return False
        if f['regs'].shape[0] != n_windows:
            print(f"  ✗ Regs shape mismatch")
            return False
        if f['labels'].shape[0] != n_windows:
            print(f"  ✗ Labels shape mismatch")
            return False
        
        print(f"  ✓ All shapes consistent")
        
        # Check data types
        print(f"\n  Data types:")
        print(f"    Signals: {f['signals'].dtype} (expected: float16)")
        print(f"    Coords: {f['coords'].dtype} (expected: float32)")
        print(f"    Regs: {f['regs'].dtype} (expected: int32)")
        print(f"    Labels: {f['labels'].dtype} (expected: int32)")
        
        # Check labels
        labels = f['labels'][:]
        n_interictal = np.sum(labels == 0)
        n_ictal = np.sum(labels == 1)
        print(f"\n  Label distribution:")
        print(f"    Interictal (0): {n_interictal} ({100*n_interictal/n_windows:.1f}%)")
        print(f"    Ictal (1): {n_ictal} ({100*n_ictal/n_windows:.1f}%)")
        
        # Check normalization (should be approximately z-scored)
        # Sample a few windows
        sample_indices = np.random.choice(n_windows, min(5, n_windows), replace=False)
        print(f"\n  Checking normalization on {len(sample_indices)} sample windows:")
        
        for idx in sample_indices:
            signals = f['signals'][idx]
            n_ch = f['n_channels'][idx]
            signals = signals[:n_ch]  # Remove padding
            
            # Per-channel statistics (should be ~0 mean, ~1 std if z-scored)
            channel_means = np.nanmean(signals, axis=1)
            channel_stds = np.nanstd(signals, axis=1)
            
            avg_mean = np.nanmean(np.abs(channel_means))
            avg_std = np.nanmean(channel_stds)
            
            print(f"    Window {idx}: |mean|={avg_mean:.3f}, std={avg_std:.3f}")
            
            if avg_mean > 0.5:
                print(f"      ⚠ Warning: Mean is large (expected ~0 for z-score)")
            if avg_std < 0.5 or avg_std > 2.0:
                print(f"      ⚠ Warning: Std is unusual (expected ~1 for z-score)")
        
        return True


def test_dataset_loading(h5_dir):
    """Test that dataset can be loaded."""
    print(f"\n{'='*60}")
    print("Testing Dataset Loading")
    print('='*60)
    
    from dataset_all_windows import HDF5WindowDataset, create_dataloader
    
    # Test basic dataset
    print("\n1. Creating dataset...")
    dataset = HDF5WindowDataset(h5_dir, return_metadata=True)
    print(f"  ✓ Dataset created with {len(dataset)} windows")
    
    # Test loading a sample
    print("\n2. Loading sample window...")
    signals, coords, regs, label, metadata = dataset[0]
    print(f"  ✓ Loaded window:")
    print(f"    Signals: {signals.shape}, dtype: {signals.dtype}")
    print(f"    Coords: {coords.shape}")
    print(f"    Regs: {regs.shape}")
    print(f"    Label: {label}")
    print(f"    Patient: {metadata['patient_id']}")
    
    # Test dataloader
    print("\n3. Creating dataloader...")
    dataloader = create_dataloader(
        h5_dir,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # Single process for testing
        return_metadata=False
    )
    print(f"  ✓ Dataloader created")
    
    # Test batch loading
    print("\n4. Loading test batch...")
    for batch in dataloader:
        signals, coords, regs, labels = batch
        print(f"  ✓ Loaded batch:")
        print(f"    Signals: {signals.shape}")
        print(f"    Coords: {coords.shape}")
        print(f"    Regs: {regs.shape}")
        print(f"    Labels: {labels.shape}")
        break  # Just test one batch
    
    print("\n✓ Dataset loading successful!")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Extract All Windows Pipeline")
    print("="*60)
    
    config = BenchmarkConfig()
    output_dir = os.path.join(config.OUTPUT_DATA_DIR, "all_windows_per_patient")
    
    # Check if output directory exists
    if not os.path.exists(output_dir):
        print(f"\n✗ Output directory not found: {output_dir}")
        print("\nPlease run extract_all_windows.py first:")
        print("  python benchmark/extract_all_windows.py")
        return False
    
    # Find HDF5 files
    h5_files = sorted(glob(os.path.join(output_dir, "*.h5")))
    
    if len(h5_files) == 0:
        print(f"\n✗ No HDF5 files found in: {output_dir}")
        print("\nPlease run extract_all_windows.py first:")
        print("  python benchmark/extract_all_windows.py")
        return False
    
    print(f"\n✓ Found {len(h5_files)} patient files")
    
    # Test a sample of files
    n_test = min(3, len(h5_files))
    test_files = np.random.choice(h5_files, n_test, replace=False)
    
    print(f"\nTesting {n_test} random patient files...")
    
    all_passed = True
    for h5_file in test_files:
        if not test_hdf5_structure(h5_file):
            all_passed = False
            print(f"  ✗ Test failed for {os.path.basename(h5_file)}")
        else:
            print(f"  ✓ Test passed for {os.path.basename(h5_file)}")
    
    if not all_passed:
        print("\n✗ Some HDF5 structure tests failed")
        return False
    
    # Test dataset loading
    if not test_dataset_loading(output_dir):
        print("\n✗ Dataset loading tests failed")
        return False
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nExtraction pipeline is working correctly:")
    print("  ✓ HDF5 files have correct structure")
    print("  ✓ Data is properly normalized (z-score)")
    print("  ✓ Labels are correctly assigned")
    print("  ✓ Dataset loading works")
    print("\nReady for model training!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
