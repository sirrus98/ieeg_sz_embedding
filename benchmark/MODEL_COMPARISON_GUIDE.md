# Model Comparison Script - Usage Guide

## Overview

The `compare_models.py` script evaluates and compares all trained benchmark models (ST, ConformerRegions, Conformer, WVNT) on validation and test sets with bootstrap confidence intervals.

## Features

- **Automatic checkpoint discovery** - Finds best checkpoints based on validation AUROC
- **Bootstrap confidence intervals** - Computes 95% CI for F1, AUROC, and AUPRC
- **Efficient data loading** - Loads dataset once and reuses for all models
- **Consistent splits** - Uses same train/val/test splits as training (70/15/15, seed=42)
- **Handles variable channels** - Properly masks padded channels for models that need it
- **CSV output** - Saves detailed results for further analysis
- **Formatted tables** - Pretty-printed comparison tables

## Quick Start

### Basic Usage

```bash
cd benchmark
python compare_models.py
```

This will:
1. Load data from default directory (from `config_benchmark.py`)
2. Find best checkpoints for all 4 models (st, conformer_regions, conformer, wvnt)
3. Evaluate on validation and test sets
4. Compute metrics with 1000 bootstrap iterations
5. Save results to `model_comparison_results.csv`
6. Print formatted comparison table

### Custom Options

```bash
# Specify data directory
python compare_models.py --data-dir data/all_windows_per_patient_global_norm

# Custom output file
python compare_models.py --output results/my_comparison.csv

# Quick test with fewer bootstraps
python compare_models.py --n-bootstrap 100

# Compare specific models only
python compare_models.py --models st wvnt

# Use max aggregation instead of mean
python compare_models.py --aggregation max

# Adjust confidence interval
python compare_models.py --confidence-interval 99

# Use different checkpoint directory
python compare_models.py --checkpoint-dir my_checkpoints
```

## Output Format

### CSV File

The script saves results to a CSV file with the following columns:

```csv
model,aggregation,split,f1_mean,f1_ci_low,f1_ci_high,auroc_mean,auroc_ci_low,auroc_ci_high,auprc_mean,auprc_ci_low,auprc_ci_high
st,mean,validation,0.8523,0.8401,0.8645,0.9234,0.9156,0.9312,0.9156,0.9078,0.9234
st,mean,test,0.8467,0.8345,0.8589,0.9178,0.9089,0.9267,0.9089,0.8998,0.9180
...
```

### Console Output

The script prints a formatted comparison table:

```
====================================================================================================
                                   Model Comparison Results
====================================================================================================

Validation Set:
----------------------------------------------------------------------------------------------------
Model                     F1 Score                       AUROC                          AUPRC                         
----------------------------------------------------------------------------------------------------
st (mean)                 0.8523 (0.8401-0.8645)         0.9234 (0.9156-0.9312)         0.9156 (0.9078-0.9234)        
conformer_regions (mean)  0.8431 (0.8309-0.8553)         0.9178 (0.9089-0.9267)         0.9089 (0.8998-0.9180)        
conformer (mean)          0.8345 (0.8223-0.8467)         0.9123 (0.9034-0.9212)         0.9034 (0.8943-0.9125)        
wvnt (mean)               0.8256 (0.8134-0.8378)         0.9067 (0.8978-0.9156)         0.8978 (0.8887-0.9069)        
----------------------------------------------------------------------------------------------------

Test Set:
... (same format)
```

## Checkpoint Discovery

The script automatically finds the best checkpoint for each model by:

1. Looking in `checkpoints/{model}_{aggregation}/best/`
2. Finding the checkpoint with highest AUROC in filename (e.g., `epoch=28-val/auroc=0.891.ckpt`)
3. Falling back to `last.ckpt` if no AUROC in filename
4. Skipping the model if no checkpoint found

### Expected Checkpoint Structure

```
checkpoints/
├── st_mean/
│   └── best/
│       ├── epoch=25-val/auroc=0.923.ckpt
│       ├── epoch=28-val/auroc=0.925.ckpt  ← This will be selected
│       └── last.ckpt
├── conformer_regions_mean/
│   └── best/
│       └── ...
├── conformer_mean/
│   └── best/
│       └── ...
└── wvnt_mean/
    └── best/
        └── ...
```

## Bootstrap Confidence Intervals

The script uses bootstrap resampling to compute confidence intervals:

1. **Resampling**: For each of N iterations (default 1000):
   - Sample with replacement from predictions/labels
   - Compute F1, AUROC, AUPRC on bootstrap sample
   
2. **Confidence Intervals**: 
   - Mean: Average across all bootstrap samples
   - CI lower: 2.5th percentile (for 95% CI)
   - CI upper: 97.5th percentile (for 95% CI)

3. **Edge Cases**:
   - Skips bootstrap samples with only one class
   - Handles metric computation failures gracefully

## Performance Optimization

The script is optimized for speed:

- **Load data once**: Dataset loaded once before model loop (~5 minutes)
- **Reuse dataloaders**: Val and test loaders created once and reused
- **No mmap**: Loads all data into RAM (faster than memory-mapped files)
- **Typical runtime**: ~5 minutes initial load + ~30 seconds per model

Expected total time for 4 models: **~7 minutes**

## Requirements

- Trained model checkpoints in `checkpoints/` directory
- NPZ data directory with preprocessed windows
- PyTorch, NumPy, Pandas, scikit-learn
- GPU recommended but not required

## Troubleshooting

### No checkpoints found

```
⚠ No checkpoint found for st_mean, skipping...
```

**Solution**: Train the model first using `train_benchmarks.py`

### Data directory not found

```
FileNotFoundError: NPZ directory not found
```

**Solution**: Specify correct data directory with `--data-dir` or check `config_benchmark.py`

### Out of memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce `--batch-size` (default is 32)

### Different splits than training

The script uses the same random seed (42) and split proportions (70/15/15) as `train_benchmarks.py`, ensuring validation and test sets match those used during training.

## Example Workflow

```bash
# 1. Train models (if not already trained)
python train_benchmarks.py --model st --aggregation mean
python train_benchmarks.py --model conformer_regions --aggregation mean
python train_benchmarks.py --model conformer --aggregation mean
python train_benchmarks.py --model wvnt --aggregation mean

# 2. Compare all models
python compare_models.py --output results/comparison_mean.csv

# 3. Quick comparison with fewer bootstraps (for testing)
python compare_models.py --n-bootstrap 100 --output results/quick_test.csv

# 4. Compare with max aggregation
python compare_models.py --aggregation max --output results/comparison_max.csv
```

## Output Analysis

After running the script, you can analyze results:

```python
import pandas as pd

# Load results
df = pd.read_csv('model_comparison_results.csv')

# Filter validation set
val_df = df[df['split'] == 'validation']

# Sort by AUROC
val_df_sorted = val_df.sort_values('auroc_mean', ascending=False)
print(val_df_sorted[['model', 'auroc_mean', 'auroc_ci_low', 'auroc_ci_high']])

# Compare test vs validation performance
for model in df['model'].unique():
    model_df = df[df['model'] == model]
    val_auroc = model_df[model_df['split'] == 'validation']['auroc_mean'].values[0]
    test_auroc = model_df[model_df['split'] == 'test']['auroc_mean'].values[0]
    print(f"{model}: Val={val_auroc:.4f}, Test={test_auroc:.4f}, Diff={val_auroc-test_auroc:.4f}")
```

## Notes

- The script properly handles models that need region information (ST, ConformerRegions, WVNT)
- Bootstrap sampling maintains class balance through random sampling
- Confidence intervals may be wider for smaller datasets or imbalanced classes
- Results are deterministic (same random seed) for reproducibility
