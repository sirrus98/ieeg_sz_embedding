# Global Normalization Guide

## Overview

The window extraction script now supports two normalization modes:

1. **Per-channel normalization** (default): Each channel is normalized using its own mean and std
   - `z = (x - mean_channel) / std_channel`
   
2. **Global normalization** (new): All channels normalized using global statistics from 200 random EDF files
   - `z = (x - mean_global) / std_global`

## Why Global Normalization?

**Benefits:**
- Preserves relative amplitude differences between channels
- More consistent scaling across patients and time points
- Can improve model generalization

**Trade-offs:**
- May not capture channel-specific dynamics as well
- Requires pre-computation of global statistics

## Usage Workflow

### Step 1: Compute Global Statistics (Run Once)

First, compute global mean and std from 200 randomly sampled EDF files:

```bash
cd benchmark
python extract_all_windows.py --compute-global-stats
```

**Output:**
- File: `data/global_normalization_stats.json`
- Contains: `global_mean`, `global_std`, `n_samples_used`, `n_files_sampled`

**Options:**
```bash
# Use more files for statistics (default: 200)
python extract_all_windows.py --compute-global-stats --n-files-for-stats 500
```

**What Happens:**
1. Scans all EDF files from all patients
2. Randomly samples 200 files (reproducible with seed=42)
3. Preprocesses each file (same pipeline as extraction)
4. Collects all signal values (before normalization)
5. Computes mean and std across ALL collected samples (vectorized)
6. Saves statistics to JSON file

**Expected Output:**
```
Computing Global Statistics from 200 Random EDF Files
============================================================

Found 2847 total EDF files
Sampling 200 files for statistics computation...

Processing files: 100%|████████████| 200/200 [15:23<00:00]

Concatenating 1,234,567,890 samples...
Computing global statistics...

✓ Global Statistics Computed:
  Mean: 0.000123
  Std: 45.678901
  Samples: 1,234,567,890
  Files: 200

✓ Saved global statistics to data/global_normalization_stats.json
```

### Step 2: Extract Windows with Global Normalization

Extract all windows using the precomputed global statistics:

```bash
python extract_all_windows.py --use-global-norm
```

**Output Directory:** `data/all_windows_per_patient_global_norm/`

**What Changes:**
- Normalization uses global mean/std instead of per-channel
- All other preprocessing steps remain identical
- Windows saved to separate directory (original data preserved)

### Step 3: Train Models with Global Norm Data

Update your training scripts to point to the new directory:

```python
# In train script
npz_dir = Path(config.OUTPUT_DATA_DIR) / "all_windows_per_patient_global_norm"
dataset = NPZWindowDataset(str(npz_dir), return_metadata=False)
```

Or for command-line scripts:

```bash
# Modify config_benchmark.py temporarily, or:
python train_benchmarks.py --model onset --data-dir all_windows_per_patient_global_norm
```

## File Structure

```
data/
├── all_windows_per_patient/              # Per-channel normalization (original)
│   ├── sub-RID0106.npz
│   ├── sub-RID0108.npz
│   └── ...
├── all_windows_per_patient_global_norm/  # Global normalization (new)
│   ├── sub-RID0106.npz
│   ├── sub-RID0108.npz
│   └── ...
└── global_normalization_stats.json       # Precomputed statistics
```

## Command Reference

### Compute Statistics

```bash
# Basic (200 files)
python extract_all_windows.py --compute-global-stats

# More files for better statistics
python extract_all_windows.py --compute-global-stats --n-files-for-stats 500

# Test computation on single patient first
python extract_all_windows.py --compute-global-stats --n-files-for-stats 10
```

### Extract Windows

```bash
# Per-channel normalization (default, original behavior)
python extract_all_windows.py

# Global normalization
python extract_all_windows.py --use-global-norm

# Custom output directory suffix
python extract_all_windows.py --use-global-norm --output-suffix "_my_experiment"
# Output: data/all_windows_per_patient_my_experiment/

# Test on single patient
python extract_all_windows.py --use-global-norm --test-patient sub-RID0106
```

## Statistics File Format

`data/global_normalization_stats.json`:

```json
{
  "global_mean": 0.00012345,
  "global_std": 45.678901,
  "n_samples_used": 1234567890,
  "n_files_sampled": 200
}
```

## Performance

### Per-Channel vs Global

| Mode | Normalization | Speed | Memory | Use Case |
|------|---------------|-------|--------|----------|
| Per-channel | Channel-specific | Baseline | Lower | Channel-specific dynamics important |
| Global | Dataset-wide | ~Same | Slightly higher (precompute) | Cross-patient consistency important |

### Vectorization

The global statistics computation is fully vectorized:
- `numpy.concatenate()` for efficient array merging
- `numpy.nanmean()` and `numpy.nanstd()` for fast computation
- No Python loops over samples

**Time Estimate:**
- Computing statistics: ~15-20 minutes (200 files, one-time)
- Extracting windows: Same time as original (no overhead after precompute)

## Comparison Example

### Per-Channel Normalization (Original)

```python
# Each channel normalized independently
channel_1: mean=0.5, std=10.2  →  z_1 = (x_1 - 0.5) / 10.2
channel_2: mean=-1.2, std=15.7 →  z_2 = (x_2 - (-1.2)) / 15.7
channel_3: mean=2.1, std=8.3   →  z_3 = (x_3 - 2.1) / 8.3
```

**Result:** Each channel has mean=0, std=1, but relative amplitudes are lost

### Global Normalization (New)

```python
# All channels use same global statistics
global: mean=0.0001, std=45.67

channel_1: z_1 = (x_1 - 0.0001) / 45.67
channel_2: z_2 = (x_2 - 0.0001) / 45.67
channel_3: z_3 = (x_3 - 0.0001) / 45.67
```

**Result:** Relative amplitudes preserved, channels may have different std

## Troubleshooting

### Issue: "Global stats file not found"

**Solution:** Run statistics computation first:
```bash
python extract_all_windows.py --compute-global-stats
```

### Issue: Statistics computation takes too long

**Solution:** Use fewer files:
```bash
python extract_all_windows.py --compute-global-stats --n-files-for-stats 100
```

### Issue: Out of memory during statistics computation

**Solution:** The code processes files one at a time and concatenates. If still OOM, reduce `n_files_for_stats`.

### Issue: Want to recompute statistics

**Solution:** Simply rerun with `--compute-global-stats`. The file will be overwritten.

### Issue: Want both normalization types

**Solution:** Extract both! They save to different directories:
```bash
# First: Per-channel (default)
python extract_all_windows.py

# Second: Global norm
python extract_all_windows.py --use-global-norm
```

## Validation

To verify global normalization worked:

```python
import numpy as np
from benchmark.dataset_all_windows import NPZWindowDataset
import json

# Load global stats
with open('data/global_normalization_stats.json') as f:
    stats = json.load(f)

print(f"Global mean: {stats['global_mean']}")
print(f"Global std: {stats['global_std']}")

# Load dataset
dataset = NPZWindowDataset('data/all_windows_per_patient_global_norm')

# Check a sample
signals, coords, regs, label = dataset[0]
print(f"\nSample mean: {signals.mean():.6f}")
print(f"Sample std: {signals.std():.6f}")
print(f"Sample min: {signals.min():.6f}")
print(f"Sample max: {signals.max():.6f}")

# Global norm should have data centered near 0 but with natural variation
```

## Recommendations

### When to Use Global Normalization:

✅ Cross-patient seizure detection  
✅ Multi-site studies  
✅ Transfer learning scenarios  
✅ When relative amplitude matters  

### When to Use Per-Channel Normalization:

✅ Single-patient analysis  
✅ Channel-specific feature learning  
✅ When absolute amplitudes don't matter  
✅ Existing baseline comparisons  

## Summary

**Quick Start:**
```bash
# 1. Compute statistics (once)
python extract_all_windows.py --compute-global-stats

# 2. Extract with global norm
python extract_all_windows.py --use-global-norm

# 3. Train models
python train_benchmarks.py --model onset  # (update data path in script)
```

**Key Benefits:**
- ✅ Preserves relative amplitudes
- ✅ Consistent scaling across patients
- ✅ Vectorized implementation (fast)
- ✅ Computed once, used forever
- ✅ Doesn't overwrite original data

**Files Created:**
- `data/global_normalization_stats.json` (statistics)
- `data/all_windows_per_patient_global_norm/*.npz` (normalized windows)
