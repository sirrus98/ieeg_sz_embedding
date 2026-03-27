"""
Analyze channel count variability in processed NPZ files vs original EDF files.

This script shows:
1. Original channel counts from EDF files
2. Channel counts after preprocessing (bipolar montage, region filtering)
3. Distribution of channels across windows
4. Which files/windows have different channel counts
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

import numpy as np
import pandas as pd
import json
from glob import glob
import mne

from config_benchmark import BenchmarkConfig


def analyze_npz_channels(npz_file):
    """Analyze channel variability within a processed NPZ file."""
    print(f"\n{'='*80}")
    print(f"Analyzing NPZ file: {os.path.basename(npz_file)}")
    print(f"{'='*80}")
    
    # Load NPZ file
    data = np.load(npz_file, mmap_mode='r', allow_pickle=True)
    
    patient_id = str(data['patient_id'])
    n_windows = data['signals'].shape[0]
    max_channels = data['signals'].shape[1]
    
    print(f"\nPatient: {patient_id}")
    print(f"Total windows: {n_windows}")
    print(f"Max channels (padded): {max_channels}")
    
    # Get actual channel counts per window
    if 'n_channels' in data:
        n_channels = data['n_channels'][:]
        print(f"\nChannel count statistics:")
        print(f"  Min channels: {n_channels.min()}")
        print(f"  Max channels: {n_channels.max()}")
        print(f"  Mean channels: {n_channels.mean():.1f}")
        print(f"  Median channels: {np.median(n_channels):.0f}")
        print(f"  Std channels: {n_channels.std():.1f}")
        
        # Show distribution
        unique_counts, counts = np.unique(n_channels, return_counts=True)
        print(f"\nChannel count distribution:")
        for ch_count, window_count in zip(unique_counts, counts):
            print(f"  {ch_count} channels: {window_count} windows ({100*window_count/n_windows:.1f}%)")
        
        # Show which files have different channel counts
        window_info = data['window_info'][:]
        ch_names = data['ch_names'][:]
        
        # Group by file
        file_channels = {}
        for i in range(n_windows):
            info = json.loads(str(window_info[i]))
            filename = info['file']
            n_ch = n_channels[i]
            
            if filename not in file_channels:
                file_channels[filename] = []
            file_channels[filename].append(n_ch)
        
        print(f"\nChannel counts by source file:")
        for filename in sorted(file_channels.keys()):
            ch_counts = file_channels[filename]
            unique_ch = set(ch_counts)
            if len(unique_ch) == 1:
                print(f"  {filename}: {list(unique_ch)[0]} channels (consistent)")
            else:
                print(f"  {filename}: {min(ch_counts)}-{max(ch_counts)} channels (VARIABLE!)")
        
        # Show example channel names for different counts
        print(f"\nExample channel names:")
        for ch_count in unique_counts[:3]:  # Show first 3 different counts
            idx = np.where(n_channels == ch_count)[0][0]
            names = json.loads(str(ch_names[idx]))
            print(f"  {ch_count} channels: {names[:5]}..." if len(names) > 5 else f"  {ch_count} channels: {names}")
    else:
        print("\nNo 'n_channels' field found - all windows use max_channels")
    
    # Labels distribution
    labels = data['labels'][:]
    print(f"\nLabel distribution:")
    print(f"  Interictal (0): {np.sum(labels == 0)} windows ({100*np.sum(labels == 0)/n_windows:.1f}%)")
    print(f"  Ictal (1): {np.sum(labels == 1)} windows ({100*np.sum(labels == 1)/n_windows:.1f}%)")


def compare_with_original_edf(patient_id, config):
    """Compare processed NPZ channels with original EDF files."""
    print(f"\n{'='*80}")
    print(f"Comparing with original EDF files: {patient_id}")
    print(f"{'='*80}")
    
    # Find all EDF files for this patient
    patient_dir = os.path.join(
        config.RAW_DATA_DIR,
        patient_id,
        "ses-clinical01",
        "ieeg"
    )
    
    if not os.path.exists(patient_dir):
        print(f"Patient directory not found: {patient_dir}")
        return
    
    edf_files = glob(os.path.join(patient_dir, "*.edf"))
    
    if len(edf_files) == 0:
        print(f"No EDF files found in {patient_dir}")
        return
    
    print(f"\nFound {len(edf_files)} EDF files")
    
    # Analyze first few EDF files
    for i, edf_file in enumerate(sorted(edf_files)[:5]):  # Show first 5
        try:
            raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
            n_orig_channels = len(raw.ch_names)
            duration = raw.times[-1]
            
            # Count how many would be bipolar
            # Simple heuristic: bipolar montage roughly halves the channels
            # (actual count depends on channel naming and types)
            
            print(f"\n  {os.path.basename(edf_file)}:")
            print(f"    Original channels: {n_orig_channels}")
            print(f"    Duration: {duration:.1f} seconds")
            print(f"    Expected ~10s windows: {int(duration / 10)}")
            print(f"    Note: Bipolar montage + region filtering reduces channel count")
            
        except Exception as e:
            print(f"\n  {os.path.basename(edf_file)}: Error - {str(e)[:100]}")


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze channel variability in processed NPZ files'
    )
    parser.add_argument(
        '--patient',
        type=str,
        default='sub-RID0106',
        help='Patient ID to analyze (default: sub-RID0106)'
    )
    parser.add_argument(
        '--compare-edf',
        action='store_true',
        help='Also compare with original EDF files'
    )
    args = parser.parse_args()
    
    config = BenchmarkConfig()
    
    # Find NPZ file for this patient
    npz_dir = os.path.join(config.OUTPUT_DATA_DIR, "all_windows_per_patient")
    npz_file = os.path.join(npz_dir, f"{args.patient}.npz")
    
    if not os.path.exists(npz_file):
        print(f"NPZ file not found: {npz_file}")
        print(f"Please run extract_all_windows.py first")
        sys.exit(1)
    
    # Analyze NPZ
    analyze_npz_channels(npz_file)
    
    # Compare with original EDF if requested
    if args.compare_edf:
        compare_with_original_edf(args.patient, config)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
