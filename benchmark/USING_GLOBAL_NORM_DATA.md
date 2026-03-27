# Using Globally Normalized Data in Training Scripts

## Quick Reference

Both `train_benchmarks.py` and `train_ictal_interictal.py` now support the `--data-dir` argument to use different normalized datasets.

---

## train_benchmarks.py

### Use Default (Per-Channel Normalization)
```bash
python train_benchmarks.py --model onset --aggregation max
```

### Use Global Normalization
```bash
python train_benchmarks.py --model onset --aggregation max --data-dir all_windows_per_patient_global_norm
```

### Full Example with Options
```bash
python train_benchmarks.py \
    --model conformer \
    --aggregation mean \
    --data-dir all_windows_per_patient_global_norm \
    --batch-size 64 \
    --learning-rate 0.001 \
    --wandb-mode online \
    --no-mmap
```

---

## train_ictal_interictal.py

### Use Default (Per-Channel Normalization)
```bash
python train_ictal_interictal.py
```

### Use Global Normalization
```bash
python train_ictal_interictal.py --data-dir all_windows_per_patient_global_norm
```

### Full Example with Options
```bash
python train_ictal_interictal.py \
    --data-dir all_windows_per_patient_global_norm \
    --batch_size 32 \
    --lr 0.0001 \
    --precision bf16-mixed \
    --wandb-mode online \
    --no-mmap
```

---

## Complete Workflow

### Step 1: Extract Data with Both Normalizations

```bash
# First: Extract with per-channel normalization (default)
python extract_all_windows.py

# Second: Compute global statistics (run once)
python extract_all_windows.py --compute-global-stats

# Third: Extract with global normalization
python extract_all_windows.py --use-global-norm
```

**Result:** Two datasets in separate directories:
- `data/all_windows_per_patient/` (per-channel norm)
- `data/all_windows_per_patient_global_norm/` (global norm)

### Step 2: Train with Per-Channel Norm (Baseline)

```bash
# Benchmark models
python train_benchmarks.py --model onset --aggregation max --wandb-mode online

# Wav2Vec2 classifier
python train_ictal_interictal.py --wandb-mode online
```

### Step 3: Train with Global Norm (Comparison)

```bash
# Benchmark models
python train_benchmarks.py \
    --model onset \
    --aggregation max \
    --data-dir all_windows_per_patient_global_norm \
    --wandb-mode online

# Wav2Vec2 classifier
python train_ictal_interictal.py \
    --data-dir all_windows_per_patient_global_norm \
    --wandb-mode online
```

### Step 4: Compare Results in W&B

Both runs will be logged to W&B with the `data_dir` field in the config, allowing easy comparison.

---

## Custom Data Directory

If you extracted data with a custom suffix:

```bash
# Extract
python extract_all_windows.py --use-global-norm --output-suffix "_experiment1"

# Train
python train_benchmarks.py --model onset --data-dir all_windows_per_patient_experiment1
```

---

## Command-Line Arguments

### train_benchmarks.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-dir` | Data directory name under OUTPUT_DATA_DIR | `all_windows_per_patient` |
| `--model` | Model to train: onset, wvnt, conformer | Required |
| `--aggregation` | Channel aggregation: mean, max | `mean` |
| `--batch-size` | Batch size | `32` |
| `--learning-rate` | Learning rate | `1e-3` |
| `--max-epochs` | Maximum epochs | `100` |
| `--wandb-mode` | W&B logging: online, offline, disabled | `online` |
| `--no-mmap` | Load all data into RAM | `False` |

### train_ictal_interictal.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-dir` | Data directory name under OUTPUT_DATA_DIR | `all_windows_per_patient` |
| `--batch_size` | Batch size (overrides config) | From config |
| `--lr` | Learning rate (overrides config) | From config |
| `--precision` | Training precision: 32, 16-mixed, bf16-mixed | `16-mixed` |
| `--wandb-mode` | W&B logging: online, offline, disabled | `online` |
| `--no-mmap` | Load all data into RAM | `False` |
| `--freeze_encoder` | Freeze encoder for feature extraction | `False` |

---

## Expected Output

When you run with `--data-dir all_windows_per_patient_global_norm`, you'll see:

```
Loading data from: C:\Users\sirrus\Desktop\ieeg_sz_embedding\data\all_windows_per_patient_global_norm
Loading NPZ dataset from directory: C:\Users\sirrus\Desktop\ieeg_sz_embedding\data\all_windows_per_patient_global_norm
  Found 106 patient files
  Total windows: 79435
  Window shape: (68, 2560)
  Interictal (0): 71166 (89.6%)
  Ictal (1): 8269 (10.4%)
  Mode: Memory-mapped (lazy loading)
✓ Loaded dataset with 79435 windows
```

---

## W&B Experiment Tracking

Both scripts now log the `data_dir` parameter to W&B, making it easy to compare experiments:

**W&B Dashboard:**
- Run 1: `data_dir: all_windows_per_patient` (per-channel norm)
- Run 2: `data_dir: all_windows_per_patient_global_norm` (global norm)

**Filtering in W&B:**
```python
# In W&B dashboard, filter by:
config.data_dir = "all_windows_per_patient_global_norm"
```

---

## Troubleshooting

### Error: "NPZ directory not found"

**Problem:** The specified data directory doesn't exist.

**Solution:** Extract the data first:
```bash
# For global norm
python extract_all_windows.py --compute-global-stats
python extract_all_windows.py --use-global-norm

# For per-channel norm
python extract_all_windows.py
```

### Which Normalization Should I Use?

**Use Per-Channel (default):**
- ✅ Channel-specific dynamics important
- ✅ Baseline experiments
- ✅ Single-patient analysis

**Use Global:**
- ✅ Cross-patient generalization important
- ✅ Relative amplitude matters
- ✅ Multi-site studies
- ✅ Transfer learning scenarios

### Can I Train on Both?

Yes! Train separate models on each dataset and compare:

```bash
# Baseline: per-channel
python train_benchmarks.py --model onset --wandb-mode online

# Experiment: global norm
python train_benchmarks.py --model onset --data-dir all_windows_per_patient_global_norm --wandb-mode online
```

Both will log to W&B with different `data_dir` tags for easy comparison.

---

## Summary

**Default behavior (no changes needed):**
```bash
python train_benchmarks.py --model onset
# Uses: data/all_windows_per_patient/
```

**To use global normalization:**
```bash
python train_benchmarks.py --model onset --data-dir all_windows_per_patient_global_norm
# Uses: data/all_windows_per_patient_global_norm/
```

**Both datasets coexist peacefully!** No overwrites, easy switching, tracked in W&B. 🎉
