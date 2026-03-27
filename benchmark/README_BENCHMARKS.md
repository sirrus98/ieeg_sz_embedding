# Benchmark Models Implementation

Three baseline models (ONSET, WVNT, Conformer) for ictal/interictal classification on variable-channel iEEG data using PyTorch Lightning.

## Architecture Overview

All models follow the same pattern:
```
Input: [batch, n_channels, 2560] 
→ Reshape: [batch*n_channels, 1, 2560]  # channels as batch
→ Univariate Model: [batch*n_channels, features]
→ Reshape: [batch, n_channels, features]
→ Aggregate (mean/max pool): [batch, features]
→ MLP Classifier: [batch, 2]
```

## Models

### 1. ONSET
- **Architecture**: Depthwise separable TCN with adaptive pooling
- **Checkpoint**: `models/ONSET/best_model.pth`
- **Features**: 32-dim after pooling
- **Works with**: 2560 timesteps (10s @ 256Hz) ✓

### 2. WVNT
- **Architecture**: WaveNet with dilated convolutions + adaptive pooling
- **Checkpoint**: `models/WVNT/wvnt_state.pth`
- **Features**: 16-dim after fc1
- **Works with**: 2560 timesteps (10s @ 256Hz) ✓ (modified)

### 3. Conformer
- **Architecture**: 1D CNN + Transformer encoder
- **Checkpoint**: None (train from scratch)
- **Features**: 40-dim embeddings
- **Works with**: 2560 timesteps (10s @ 256Hz) ✓ (new 1D version)

## Installation

```bash
pip install pytorch-lightning>=2.0.0
pip install torchmetrics>=0.11.0
pip install tensorboard>=2.12.0
pip install wandb>=0.15.0
pip install einops>=0.6.0
pip install scikit-learn seaborn matplotlib pandas
```

## Quick Start

### 1. Extract Windows (if not done)

```bash
cd c:\Users\sirrus\Desktop\ieeg_sz_embedding
.venv\Scripts\python.exe benchmark\extract_all_windows.py
```

This creates `data/all_windows_per_patient/*.npz` files.

### 2. Train a Model

**Basic training:**
```bash
python benchmark/train_benchmarks.py --model onset --aggregation mean
```

**With options:**
```bash
python benchmark/train_benchmarks.py \
    --model wvnt \
    --aggregation max \
    --batch-size 64 \
    --max-epochs 50 \
    --learning-rate 0.001 \
    --freeze-backbone \
    --checkpoint-every-n-steps 5000 \
    --wandb-mode offline
```

**All arguments:**
- `--model`: onset, wvnt, conformer
- `--aggregation`: mean, max
- `--batch-size`: default 32
- `--max-epochs`: default 100
- `--learning-rate`: default 1e-3
- `--weight-decay`: default 1e-4
- `--num-workers`: default 0 (use 0 for Windows)
- `--freeze-backbone`: freeze pretrained weights
- `--checkpoint-every-n-steps`: default 3000
- `--wandb-mode`: online, offline, disabled

### 3. Monitor Training

**TensorBoard:**
```bash
tensorboard --logdir lightning_logs
# Open browser to http://localhost:6006
```

**Weights & Biases:**
```bash
wandb login  # First time only
# Visit https://wandb.ai/your-username/ieeg-seizure-detection
```

**CSV Logs:**
```python
import pandas as pd
metrics = pd.read_csv('csv_logs/onset_mean/version_0/metrics.csv')
print(metrics[['epoch', 'train/loss', 'val/acc', 'val/auroc']])
```

### 4. Evaluate Model

```bash
python benchmark/evaluate_benchmarks.py \
    --checkpoint checkpoints/onset_mean/best/epoch=28-val_auroc=0.891.ckpt \
    --output-dir evaluation_results/onset_mean
```

This generates:
- `metrics.csv` - Overall performance
- `confusion_matrix.png` - Confusion matrix
- `roc_curve.png` - ROC curve
- `precision_recall_curve.png` - PR curve
- `per_patient_results.csv` - Per-patient breakdown
- `all_predictions.csv` - All predictions
- `classification_report.txt` - Detailed report

## File Structure

```
benchmark/
├── models/
│   ├── ONSET/
│   │   ├── model.py (modified: added get_features)
│   │   └── best_model.pth
│   ├── WVNT/
│   │   ├── model.py (original)
│   │   ├── model_adaptive.py (new: adaptive pooling)
│   │   └── wvnt_state.pth
│   ├── conformer/
│   │   ├── conformer.py (original 2D)
│   │   └── conformer_1d.py (new: 1D version)
│   └── wrappers/
│       ├── __init__.py
│       ├── base_wrapper.py
│       ├── onset_wrapper.py
│       ├── wvnt_wrapper.py
│       └── conformer_wrapper.py
├── lightning_modules/
│   ├── __init__.py
│   └── ictal_classifier.py
├── train_benchmarks.py
├── evaluate_benchmarks.py
├── dataset_all_windows.py
└── config_benchmark.py
```

## Checkpointing

### Two-Tier System

**Best Checkpoints** (validation AUROC):
- Location: `checkpoints/{model}_{aggregation}/best/`
- Saves top 3 + last
- Use for inference

**Periodic Checkpoints** (every N steps):
- Location: `checkpoints/{model}_{aggregation}/periodic/`
- Saves all checkpoints
- Use for resuming training

### Resume Training

```python
python benchmark/train_benchmarks.py \
    --model onset \
    --aggregation mean \
    --resume-from checkpoints/onset_mean/periodic/step=9000-epoch=4.ckpt
```

## Expected Performance

Training on ~645 windows from 1 patient (sub-RID0106):
- **Training time**: ~1-2 minutes per epoch (GPU)
- **Memory**: ~2-4 GB GPU RAM
- **Convergence**: ~20-30 epochs

Full dataset (all patients):
- **Total windows**: Tens of thousands
- **Training time**: Several hours (GPU)

## Metrics Tracked

- **Accuracy**: Overall classification accuracy
- **AUROC**: Area under ROC curve
- **F1 Score**: Harmonic mean of precision/recall
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)

## Troubleshooting

### Windows Multiprocessing Error
Use `--num-workers 0` in training script.

### CUDA Out of Memory
Reduce `--batch-size` (try 16 or 8).

### W&B Login Issues
Use `--wandb-mode offline` or `--wandb-mode disabled`.

### Checkpoint Not Found
Ensure paths are correct. Use `best_model.pth` for ONSET, `wvnt_state.pth` for WVNT.

## Citation

If you use these models, please cite:
- ONSET: Lightweight Seizure Detector paper
- WVNT: WaveNet-based detection paper
- Conformer: EEG Conformer paper

## License

See individual model directories for license information.
