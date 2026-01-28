# Seizure Classification Benchmarks

This directory contains two supervised classification benchmarks for comparing the MultivarWav2Vec2 model against SOTA methods:

1. **Semiology Classification** - Multi-class classification of seizure types
2. **Ictal vs Interictal Detection** - Binary classification for seizure detection

## 📁 Directory Structure

```
benchmark/
├── config_benchmark.py              # Configuration and hyperparameters
├── model_classifier.py              # Classifier model with encoder + classification head
│
├── dataset_semiology.py             # Dataset for semiology classification
├── train_semiology.py               # Training script for semiology
│
├── extract_interictal_data.py       # Extract non-seizure windows from raw data
├── dataset_ictal_interictal.py      # Dataset for binary classification
├── train_ictal_interictal.py        # Training script for binary classification
│
├── extract_embeddings.py            # Extract embeddings for SOTA comparison
└── README.md                         # This file
```

## 🚀 Quick Start

### Step 1: Configure Paths

Edit `config_benchmark.py` and update the paths to match your environment:

```python
DATA_DIR = "/path/to/your/data"
RAW_DATA_DIR = "/path/to/raw/bids/data"
PRETRAINED_CKPT_DIR = "/path/to/pretrained/checkpoints"
```

### Step 2: Extract Interictal Data (for Binary Classification)

Before running the ictal/interictal benchmark, extract non-seizure windows:

```bash
cd benchmark
python extract_interictal_data.py
```

This will create `all_interictal_coords_data_spatiotemporal_regs_10s.pkl` in your data directory.

### Step 3: Train Models

#### Option A: Semiology Classification

```bash
# Train from pretrained weights (recommended)
python train_semiology.py --pretrained_ckpt /path/to/pretrained.ckpt

# Train from scratch
python train_semiology.py

# Feature extraction mode (freeze encoder)
python train_semiology.py --pretrained_ckpt /path/to/pretrained.ckpt --freeze_encoder
```

#### Option B: Ictal vs Interictal Detection

```bash
# Train from pretrained weights (recommended)
python train_ictal_interictal.py --pretrained_ckpt /path/to/pretrained.ckpt

# Train from scratch
python train_ictal_interictal.py

# With custom hyperparameters
python train_ictal_interictal.py --batch_size 64 --lr 1e-3
```

### Step 4: Extract Embeddings

After training, extract embeddings for comparison with other models:

```bash
# Semiology embeddings
python extract_embeddings.py semiology /path/to/checkpoint.ckpt

# Ictal/Interictal embeddings
python extract_embeddings.py ictal_interictal /path/to/checkpoint.ckpt
```

Embeddings are saved as `.npz` files in the `benchmark_embeddings/` directory.

## 📊 Using Extracted Embeddings

Load embeddings in Python for comparison with other models:

```python
import numpy as np

# Load embeddings
data = np.load('benchmark_embeddings/semiology_embeddings.npz')

# Access data
train_embeddings = data['train_embeddings']  # (N, embedding_dim)
train_labels = data['train_labels']          # (N,)
train_patient_ids = data['train_patient_ids'] # (N,)
label_names = data['label_names']            # List of class names

# Similarly for val and test
val_embeddings = data['val_embeddings']
test_embeddings = data['test_embeddings']

# Number of classes
num_classes = data['num_classes']
```

## 📝 Task Details

### Task 1: Semiology Classification

**Objective**: Classify seizure types based on clinical semiology

**Data**: 10-second ictal (seizure onset) windows with semiology labels from clinician annotations

**Metrics**:
- Accuracy
- Macro F1-score
- Per-class Precision/Recall
- Confusion Matrix

**Notes**:
- Classes with fewer than `MIN_SAMPLES_PER_CLASS` (default: 10) are grouped as "Other"
- Class weights are applied to handle imbalance
- Patient-stratified splits ensure no patient overlap between train/val/test

### Task 2: Ictal vs Interictal Detection

**Objective**: Binary classification to detect seizure vs non-seizure periods

**Data**: 
- Ictal: 10-second windows from seizure onset (label=1)
- Interictal: 10-second windows from non-seizure periods (label=0)
  - Extracted with ≥1 hour buffer from any seizure to avoid contamination

**Metrics**:
- Accuracy
- F1-score
- Precision/Recall
- ROC-AUC
- Confusion Matrix

**Notes**:
- This is a standard benchmark task in seizure detection literature
- Results are directly comparable to published SOTA methods
- Patient-stratified splits prevent data leakage

## ⚙️ Configuration Options

Key hyperparameters in `config_benchmark.py`:

```python
# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4              # Lower for fine-tuning
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Data splits (patient-stratified)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Semiology specific
MIN_SAMPLES_PER_CLASS = 10
USE_CLASS_WEIGHTS = True

# Ictal/Interictal specific
INTERICTAL_BUFFER_TIME = 3600     # 1 hour buffer
INTERICTAL_WINDOWS_PER_PATIENT = 2
```

## 🔧 Model Architecture

The classifier consists of:

1. **Encoder**: MultivarWav2Vec2 (from `../code/model_v5.py`)
   - Feature extraction with 1D convolutions
   - Spatiotemporal positional encoding with brain region information
   - Spatial transformer (across channels/regions)
   - Temporal transformer (across time frames)
   - CLS token for sequence-level representation

2. **Classification Head**:
   ```
   Linear(embedding_dim -> embedding_dim // 2)
   ReLU + Dropout(0.3)
   Linear(embedding_dim // 2 -> num_classes)
   ```

3. **Training Modes**:
   - **Fine-tuning** (default): Train entire model end-to-end
   - **Feature extraction**: Freeze encoder, train only classification head

## 📈 Monitoring Training

Training logs are saved to `benchmark_logs/`:

```bash
# View training logs
tensorboard --logdir benchmark_logs/
```

Model checkpoints are saved to `benchmark_checkpoints/`:
- `semiology/` - Semiology classification models
- `ictal_interictal/` - Binary classification models

## 🎯 Expected Results

### Semiology Classification
- Moderate difficulty due to class imbalance and subtle clinical differences
- Expected accuracy: 40-60% (depends on number of classes)
- Macro F1: 35-55%

### Ictal vs Interictal Detection
- Easier task with clearer signal differences
- Expected accuracy: 85-95%
- Expected F1: 85-95%
- Expected AUROC: 90-98%

## 🐛 Troubleshooting

### "Interictal data not found"
Run `python extract_interictal_data.py` first to create interictal dataset.

### "No module named 'config'"
The scripts expect a `config.py` file in `../code/`. If missing, the datasets will fail to load metadata. Ensure the main project config exists.

### CUDA out of memory
Reduce `BATCH_SIZE` in `config_benchmark.py` or pass `--batch_size 16` to training scripts.

### Different number of classes
The number of semiology classes depends on your data and `MIN_SAMPLES_PER_CLASS`. Adjust this parameter if you want more/fewer classes.

## 📚 Comparison with Other Models

To compare with other SOTA models:

1. **Extract embeddings** from both models using the same train/val/test splits
2. **Train a linear classifier** on top of embeddings:
   ```python
   from sklearn.linear_model import LogisticRegression
   
   # Train on embeddings
   clf = LogisticRegression(max_iter=1000)
   clf.fit(train_embeddings, train_labels)
   
   # Test
   accuracy = clf.score(test_embeddings, test_labels)
   ```

3. **Compare metrics** directly since splits are patient-stratified and reproducible

## 📖 Citation

If you use this benchmark, please cite:

```bibtex
@article{yourpaper2024,
  title={Your Paper Title},
  author={Your Name et al.},
  journal={Journal Name},
  year={2024}
}
```

## 📧 Contact

For questions or issues, please contact [your email] or open an issue on GitHub.

## 🔗 Related Files

- Main model: `../code/model_v5.py`
- Dataset creation: `../code/make_dataset.py`
- Self-supervised training: `../code/train.py`
- Data visualization: `../code/data_visualization.ipynb`
