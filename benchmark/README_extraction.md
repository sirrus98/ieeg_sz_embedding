# All Windows Extraction Pipeline

This directory contains code for extracting all 10-second windows from iEEG data following the exact preprocessing pipeline from `code/make_dataset.py`.

## Overview

The extraction pipeline:
1. **Extracts all non-overlapping 10-second windows** from both ictal and interictal EDF files
2. **Follows the exact preprocessing** from `make_dataset.py`:
   - Notch filter (60Hz)
   - Bandpass filter (0.5-120Hz, order 10)
   - Bipolar montage
   - Resampling to 256Hz
   - Per-channel z-score normalization
3. **Labels windows** based on timing:
   - Ictal files: pre-ictal (0-30s) → label 0, ictal (30s-offset) → label 1, post-ictal → label 0
   - Interictal files: all windows → label 0
4. **Saves to HDF5** (one file per patient) for efficient lazy loading

## Files

### Core Scripts

- **`extract_all_windows.py`** - Main extraction script
  - Loads EDF files
  - Applies preprocessing pipeline
  - Extracts all windows
  - Saves to HDF5 format

- **`dataset_all_windows.py`** - PyTorch dataset for loading windows
  - Lazy loading from HDF5
  - Handles variable channel counts with padding
  - Batch collation with proper padding

- **`test_extraction.py`** - Testing and validation
  - Verifies HDF5 structure
  - Checks normalization
  - Tests dataset loading

### Configuration

- **`config_benchmark.py`** - Paths and parameters
  - Update paths for your environment
  - Set data directories
  - Configure extraction parameters

## Usage

### 1. Configure Paths

Edit `config_benchmark.py` and update:
```python
DATA_DIR = "/path/to/data"  # Read-only data from make_dataset.py
OUTPUT_DATA_DIR = "/path/to/output"  # Your writable directory
RAW_DATA_DIR = "/path/to/bids/data"  # Raw EDF files
```

### 2. Extract Windows

```bash
cd /users/zcxu/ieeg_sz_embedding
python benchmark/extract_all_windows.py
```

This will:
- Process all patients one at a time (memory efficient)
- Save one HDF5 file per patient to `OUTPUT_DATA_DIR/all_windows_per_patient/`
- Print statistics about extraction

**Expected output:**
```
Extracting windows from 102 patients...
Patients: 100%|██████████| 102/102

✓ Extracted 50,000 total windows
  Patients processed: 102
  Interictal (label=0): 45,000
  Ictal (label=1): 5,000
```

### 3. Test Extraction

```bash
python benchmark/test_extraction.py
```

This verifies:
- HDF5 files have correct structure
- Data is properly normalized
- Labels are correct
- Dataset loading works

### 4. Use in Training

```python
from benchmark.dataset_all_windows import create_dataloader

# Create dataloader
dataloader = create_dataloader(
    h5_dir='path/to/all_windows_per_patient',
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Training loop
for signals, coords, regs, labels in dataloader:
    # signals: (batch, max_channels, 2560)
    # coords: (batch, max_channels, 3)
    # regs: (batch, max_channels)
    # labels: (batch,) - 0=interictal, 1=ictal
    
    # Your model training code here
    pass
```

## Data Format

### HDF5 Structure (per patient file)

```
sub-RID0106.h5
├── signals       (n_windows, max_channels, 2560)  float16
├── coords        (n_windows, max_channels, 3)     float32
├── regs          (n_windows, max_channels)        int32
├── n_channels    (n_windows,)                     int32
├── labels        (n_windows,)                     int32
├── patient_id    string
├── ch_names      (n_windows,)                     JSON strings
└── window_info   (n_windows,)                     JSON metadata
```

### Labels

- `0` - Interictal (non-seizure)
  - From interictal files: all windows
  - From ictal files: pre-ictal (0-30s) and post-ictal periods
- `1` - Ictal (seizure)
  - From ictal files: during seizure period (30s to offset)

### Normalization

Data is **already normalized** during extraction using per-channel z-score:
```python
normalized = (x - nanmean(x)) / nanstd(x)
```

No additional normalization should be applied in the dataset/dataloader.

## Preprocessing Pipeline

The extraction follows the **exact** preprocessing from `code/make_dataset.py`:

1. Load EDF: `mne.io.read_raw_edf()`
2. Clean labels: `clean_labels(ch_names, pt=None)`
3. Channel types: `check_channel_types(ch_names)`
4. Notch filter: `notch_filter(signals.T, fs).T` (60Hz)
5. Bandpass: `bandpass_filter(signals, fs, order=10, lo=0.5, hi=120)`
6. Bipolar montage: `bipolar_montage(signals, ch_types)`
7. Resample: `resample_poly()` to 256Hz
8. Match annotations: Filter to `master_bipolars`
9. Filter regions: Keep only valid regions from `unique_regs.csv`
10. **Normalize**: Per-channel z-score `(x - nanmean(x)) / nanstd(x)`
11. Convert: `astype(np.float16)` to save space

## Memory Efficiency

The pipeline is designed to handle large datasets:
- **Process one patient at a time** - avoids loading all data into memory
- **Save immediately** - frees memory after each patient
- **Lazy loading** - dataset loads windows on-demand
- **Compression** - HDF5 uses gzip compression

## Troubleshooting

### Import errors
```
ImportError: cannot import name 'clean_labels'
```
**Solution:** The scripts add `code/` to sys.path. Make sure `code/utils.py` exists.

### Permission errors
```
PermissionError: [Errno 13] Permission denied
```
**Solution:** Update `OUTPUT_DATA_DIR` in `config_benchmark.py` to a directory you can write to.

### Memory errors
```
MemoryError: Unable to allocate array
```
**Solution:** The pipeline processes one patient at a time, so this shouldn't happen. Check system memory.

### No windows extracted
```
✓ Extracted 0 total windows
```
**Solution:** Check that:
- `RAW_DATA_DIR` points to correct BIDS directory
- `DATA_DIR` has `metadata/` and `atlases/` subdirectories
- EDF files exist for patients in metadata

## Performance

Typical extraction time:
- **~1-2 seconds per patient** on average
- **~3-5 minutes for 102 patients**
- Output size: **~1-5 GB** total (depends on number of windows)

## Validation

To verify the extraction matches the original preprocessing:

1. **Check normalization**: Mean should be ~0, std should be ~1
   ```python
   python benchmark/test_extraction.py
   ```

2. **Compare with original**: Load a window from both sources
   ```python
   # Original (make_dataset.py output)
   with open('all_sz_coords_data_spatiotemporal_regs_10s.pkl', 'rb') as f:
       orig_data = pickle.load(f)
   
   # New extraction
   with h5py.File('sub-RID0106.h5', 'r') as f:
       new_data = f['signals'][0]
   
   # Should have similar statistics
   ```

## Citation

If you use this extraction pipeline, please cite the original work that developed the preprocessing:
```
[Citation for ieeg_sz_embedding project]
```
