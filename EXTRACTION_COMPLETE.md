# Window Extraction Complete ✓

## Summary

Successfully rewrote the window extraction pipeline to follow `code/make_dataset.py` preprocessing exactly and extract ALL 10-second windows from ictal and interictal files.

## What Was Implemented

### 1. Complete Rewrite of `benchmark/extract_all_windows.py`

**Preprocessing Pipeline (Exact Match to make_dataset.py):**
1. Load EDF with `mne.io.read_raw_edf()`
2. Clean channel labels with `clean_labels()`
3. Get channel types with `check_channel_types()`
4. Apply 60Hz notch filter with `notch_filter()`
5. Apply bandpass filter (0.5-120Hz, order=10) with `bandpass_filter()`
6. Apply bipolar montage with `bipolar_montage()`
7. Resample to 256Hz with `resample_poly()`
8. Per-channel z-score normalization: `(x - nanmean(x)) / nanstd(x)`
9. Match channels to `master_bipolars` annotations
10. Filter to valid regions from `unique_regs.csv`
11. Convert to float16

**Window Extraction Logic:**
- **Ictal files**: Extract ALL non-overlapping 10-second windows:
  - Pre-ictal (0-30s): labeled as interictal (0)
  - Ictal period (30s to seizure offset): labeled as ictal (1)
  - Post-ictal (after offset): labeled as interictal (0)
- **Interictal files**: Extract ALL non-overlapping 10-second windows (label=0)

**Variable Channel Handling:**
- Different EDF files for the same patient may have different electrode configurations
- Pad to `max_channels` across all windows for that patient
- Store actual channel count in `n_channels` array
- All electrodes mapped to brain regions via `master_bipolars`

### 2. NPZ File Format (Per-Patient)

Each patient gets one `.npz` file with:
```python
{
    'signals': (n_windows, max_channels, 2560),  # float16, padded
    'coords': (n_windows, max_channels, 3),      # float32, padded
    'regs': (n_windows, max_channels),           # int32, padded
    'n_channels': (n_windows,),                   # int32, actual channel count
    'labels': (n_windows,),                       # int32 (0=interictal, 1=ictal)
    'patient_id': str,
    'ch_names': (n_windows,) object array of JSON strings,
    'window_info': (n_windows,) object array with metadata
}
```

### 3. Updated `benchmark/dataset_all_windows.py`

**Features:**
- Memory-mapped NPZ loading with `np.load(mmap_mode='r')`
- Lazy loading per window (only reads from disk when accessed)
- Handles variable channel counts with padding
- Custom collate function for batch loading
- Compatible with PyTorch DataLoader

## Test Results

**Test Patient: sub-RID0106**
- ✓ Total windows extracted: 645
  - Ictal (label=1): 130 (20.2%)
  - Interictal (label=0): 515 (79.8%)
- ✓ Max channels: 94
- ✓ File size: 270.8 MB (compressed NPZ)
- ✓ Window shape: (94, 2560) = 94 channels × 2560 timesteps (10s @ 256Hz)
- ✓ Data properly normalized: std ≈ 1.0 (z-score normalization)
- ✓ Dataset loader working correctly
- ✓ Batch loading working correctly

## Usage

### Extract All Windows for All Patients

```bash
cd c:\Users\sirrus\Desktop\ieeg_sz_embedding
.venv\Scripts\python.exe benchmark\extract_all_windows.py
```

This will:
- Process all 135 patients
- Extract thousands of windows per patient
- Save one compressed NPZ file per patient to `data/all_windows_per_patient/`

### Test on Single Patient

```bash
.venv\Scripts\python.exe benchmark\extract_all_windows.py --test-patient sub-RID0106
```

### Load Data in Training Script

```python
from benchmark.dataset_all_windows import NPZWindowDataset, create_dataloader

# Create dataset
dataset = NPZWindowDataset(
    npz_dir_path="data/all_windows_per_patient",
    return_metadata=False
)

# Create dataloader
dataloader = create_dataloader(
    npz_dir_path="data/all_windows_per_patient",
    batch_size=32,
    shuffle=True,
    num_workers=0,  # Use 0 on Windows to avoid multiprocessing issues
    pin_memory=True
)

# Use in training loop
for signals, coords, regs, labels in dataloader:
    # signals: (batch, max_channels, 2560)
    # coords: (batch, max_channels, 3)
    # regs: (batch, max_channels) - region indices
    # labels: (batch,) - 0=interictal, 1=ictal
    pass
```

## Key Improvements

1. ✓ **Exact preprocessing match** to `make_dataset.py` - ensures model compatibility
2. ✓ **All windows extracted** - no data left behind
3. ✓ **Variable channels handled** - different electrode configurations per file OK
4. ✓ **Memory efficient** - saves after processing, no pre-allocation needed
5. ✓ **Compressed storage** - NPZ with compression saves disk space
6. ✓ **Lazy loading** - memory-mapped NPZ for efficient training
7. ✓ **Rich metadata** - patient ID, channel names, window info preserved

## Files Modified

- ✓ `benchmark/extract_all_windows.py` - Complete rewrite (636 lines)
- ✓ `benchmark/dataset_all_windows.py` - Minor fix (num_workers=0 in test)

## Next Steps

Run the full extraction on all patients:

```bash
.venv\Scripts\python.exe benchmark\extract_all_windows.py
```

Expected output:
- ~135 NPZ files (one per patient)
- Tens of thousands of windows total
- Ready for model training with proper preprocessing
