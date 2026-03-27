# Linear Evaluation Experiment

## Overview

This directory contains scripts to replicate the linear evaluation experiment from `code/test_nb copy.ipynb` (cells 99-103) using trained WVNT and Conformer models.

## Files

### Scripts

1. **`extract_embeddings.py`** - Extract per-seizure embeddings from trained models
   - Loads 888 seizures from `data/all_sz_coords_data_spatiotemporal_regs_10s.pkl`
   - Loads trained checkpoints from `checkpoints/{model}_{aggregation}/best/`
   - Extracts embeddings using `get_embeddings()` method (features after aggregation, before classification)
   - Computes SOZ-based labels from `data/metadata/soz_electrodes.csv`
   - Saves embeddings and labels to `benchmark/embeddings/{model}_{aggregation}_embeddings.npz`

2. **`linear_evaluation.py`** - Run 10-fold cross-validation with 2-layer MLP
   - Loads embeddings from npz file
   - Trains 2-layer MLP classifier (input_dim → input_dim → 4 classes)
   - Reports balanced accuracy, accuracy, F1, precision, recall
   - Saves confusion matrix and metrics to `benchmark/results/linear_eval/`

### Model Modifications

Modified wrappers to add `get_embeddings()` methods:
- `models/wrappers/wvnt_wrapper.py` - Extract 256-dim features after channel aggregation
- `models/wrappers/conformer_wrapper.py` - Extract emb_size-dim features after class token
- `models/conformer/conformer_1d.py` - Added `get_embeddings()` to base model

## Usage

### Step 1: Extract Embeddings

```bash
cd benchmark
python extract_embeddings.py
```

This will create:
- `embeddings/wvnt_mean_embeddings.npz`
- `embeddings/conformer_mean_embeddings.npz`

Each file contains:
- `embeddings`: [n_seizures, embed_dim] array
- `labels`: [n_seizures] string array ("Left Frontal", "Right Frontal", "Left Temporal", "Right Temporal")
- `patient_ids`: [n_seizures] patient ID array
- `seizure_ids`: [n_seizures] seizure index array

### Step 2: Run Linear Evaluation

```bash
# WVNT evaluation
python linear_evaluation.py embeddings/wvnt_mean_embeddings.npz --output-dir results/linear_eval --epochs 500

# Conformer evaluation
python linear_evaluation.py embeddings/conformer_mean_embeddings.npz --output-dir results/linear_eval --epochs 500
```

Options:
- `--output-dir`: Directory to save results
- `--epochs`: Training epochs per fold (default: 500)
- `--n-splits`: Number of CV folds (default: 10)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)

### Quick Test (fewer epochs)

For faster testing:

```bash
python linear_evaluation.py embeddings/wvnt_mean_embeddings.npz --epochs 50
```

## Results

Results are saved to `benchmark/results/linear_eval/`:
- `{model}_confusion_matrix.png` - Confusion matrix visualization
- `{model}_metrics.txt` - Text file with all metrics

Console output includes:
- Balanced Accuracy
- Accuracy
- Per-class precision, recall, F1
- Confusion matrix

## Implementation Details

### SOZ Label Derivation

Labels are computed from clinical seizure onset zone (SOZ) annotations:

1. Load SOZ electrodes from `data/metadata/soz_electrodes.csv`
2. For each seizure, identify channels marked as SOZ
3. Map SOZ channels to brain regions using `data/unique_regs.csv`
4. Count regions by category:
   - Left Frontal: "Frontal" in region name and "L" in name
   - Right Frontal: "Frontal" in region name and "R" in name
   - Left Temporal: "Temporal" in region name and "L" in name
   - Right Temporal: "Temporal" in region name and "R" in name
5. Assign label based on maximum count

### Dataset Statistics

From extraction run:
- Total seizures loaded: 888
- Seizures with metadata: 878
- Seizures with SOZ labels: 771
- Label distribution:
  - Left Frontal: 408 (53%)
  - Right Frontal: 67 (9%)
  - Left Temporal: 263 (34%)
  - Right Temporal: 33 (4%)

### Model Specifications

**WVNT:**
- Embedding dimension: 256
- Extracted after: Masked channel aggregation (mean)
- Checkpoint: `checkpoints/wvnt_mean/best/last.ckpt`

**Conformer:**
- Embedding dimension: 32
- Extracted after: ViT-style class token (normalized)
- Checkpoint: `checkpoints/conformer_mean/best/last-v1.ckpt`

### Linear Classifier Architecture

Replicates notebook cell 101:

```python
class LinearClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes=4, learning_rate=1e-3):
        self.linear_1 = nn.Linear(input_dim, input_dim)
        self.linear = nn.Linear(input_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
```

Training:
- 10-fold cross-validation
- 500 epochs per fold
- Batch size: 32
- Optimizer: Adam (lr=1e-3)
- No data augmentation
- No dropout or regularization

## Comparison with Notebook

This implementation is a 1:1 replication of the notebook experiment:

| Component | Notebook | This Implementation |
|-----------|----------|---------------------|
| Model | SWaV (self-supervised) | WVNT & Conformer (supervised ictal classification) |
| Embedding extraction | `model.get_features()` | `model.get_embeddings()` |
| Label derivation | SOZ metadata (cell 99) | Same (SOZ metadata) |
| Classifier | 2-layer MLP | Same (2-layer MLP) |
| Training | 10-fold CV, 500 epochs | Same |
| Metrics | Balanced accuracy, confusion matrix | Same |

## Expected Runtime

- Embedding extraction: ~2 minutes
- Linear evaluation per model: ~10-15 minutes (10 folds × 500 epochs)
- Total: ~30 minutes for both models

## Troubleshooting

### "No checkpoint found"
- Ensure checkpoints exist in `checkpoints/{model}_mean/best/`
- Script looks for versioned checkpoints (last-v2.ckpt > last-v1.ckpt > last.ckpt)

### "Indices out of bounds"
- Ensure using `data/gui_data/master_metadata.csv` (878 rows matching the seizures)
- Not the root `data/master_metadata.csv` (different structure)

### "Boolean index mismatch"
- Fixed in current version - uses actual channel counts, not max_channels

### GPU memory issues
- Reduce `--batch-size` (default: 32)
- Extraction uses batch_size=32 by default

## Notes

- Only seizures with SOZ annotations are included (771 out of 878)
- Class imbalance: Left Frontal dominates (53%), Right Temporal is rare (4%)
- This may affect balanced accuracy vs regular accuracy
- Models are evaluated in `eval()` mode with trained weights frozen
