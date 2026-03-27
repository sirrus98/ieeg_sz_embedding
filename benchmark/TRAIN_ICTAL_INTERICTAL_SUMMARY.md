# Train Ictal/Interictal Classification - Implementation Summary

## Overview

Successfully rewrote `benchmark/train_ictal_interictal.py` to use the NPZ dataset format with `MultivarWav2Vec2Classifier` model for seizure detection (ictal vs interictal classification).

## Key Changes

### 1. Dataset Integration
- **Replaced**: Old `IctalInterictalDataset` 
- **New**: `NPZWindowDataset` from `data/all_windows_per_patient`
- **Benefits**:
  - Memory-mapped loading (efficient for large datasets)
  - Handles 79,435 windows from 106 patients
  - Class distribution: 89.6% interictal, 10.4% ictal
  - Optimized class weight calculation (reuses counts from initialization)

### 2. Batch Format Update
- **Old batch format**: `(signals, regs, labels, metadata)`
- **New batch format**: `(signals, coords, regs, labels)`
- Updated all Lightning module steps (training, validation, test)

### 3. Custom Collate Function
```python
def collate_fn(batch):
    """Pads all samples to 150 channels (channel_buffer_size)"""
```
- Handles variable channel counts (68-107 channels in data)
- Pads to fixed 150 channels to match model architecture
- Ensures compatibility with `MultivarWav2Vec2Classifier` temporal transformer

### 4. Model Architecture
- **Model**: `MultivarWav2Vec2Classifier`
- **Input**: `(batch, 150, 2560)` - 150 channels × 2560 timesteps (10 seconds @ 256 Hz)
- **Output**: `(batch, 2)` - binary classification logits
- **Parameters**: 467M trainable parameters
- **Features**:
  - Conv1D feature encoder
  - Spatial transformer (across channels)
  - Temporal transformer (across time) with CLS token
  - Classification head

### 5. Checkpoint Loading
- **Default checkpoint**: `C:\Users\sirrus\Desktop\ieeg_sz_embedding\checkpoints\checkpoints_10s_128_batch\model_epoch-epoch=19-train_loss=5.56027.ckpt`
- Supports loading from pretrained weights with `strict=False`
- Graceful fallback if checkpoint not found

### 6. Mixed Precision Training
- **Default**: `16-mixed` (FP16 automatic mixed precision)
- **Options**: `32` (FP32), `16-mixed` (FP16), `bf16-mixed` (BF16)
- Speeds up training and reduces memory usage

### 7. Lightning Module Features
- **Metrics**: Accuracy, F1, Precision, Recall, AUROC, Confusion Matrix
- **Optimizer**: AdamW (default LR: 1e-4)
- **Scheduler**: ReduceLROnPlateau
- **Class weights**: Automatic calculation from dataset for imbalanced data
- **Fixed**: Removed deprecated `verbose` parameter from scheduler

## Command Line Arguments

```bash
python train_ictal_interictal.py [OPTIONS]

Options:
  --batch_size INT           Batch size (default: 16)
  --lr FLOAT                 Learning rate (default: 1e-4)
  --pretrained_ckpt PATH     Path to checkpoint (default: see above)
  --freeze_encoder           Freeze encoder for feature extraction
  --precision STR            Training precision: 32, 16-mixed, bf16-mixed
```

## Training Configuration

From `config_benchmark.py`:
- **Batch size**: 16
- **Learning rate**: 1e-4
- **Max epochs**: 100
- **Early stopping**: patience=10
- **Train/Val/Test split**: 70/15/15
- **Class weights**: [0.559, 4.804] (compensates for 89.6/10.4 imbalance)

## Data Flow

```
NPZ Files (data/all_windows_per_patient/*.npz)
    ↓
NPZWindowDataset (memory-mapped loading)
    ↓
Random Split (70/15/15)
    ↓
DataLoader (collate_fn: pad to 150 channels)
    ↓
IctalInterictalClassifierModule (Lightning)
    ↓
MultivarWav2Vec2Classifier (467M params)
    ↓
Binary Classification (interictal vs ictal)
```

## Test Suite

Created `benchmark/test_train_ictal_interictal.py` with 4 comprehensive tests:

1. **test_dataset_loading**: ✓ PASSED
   - Verifies NPZ loading, splits, and batch shapes
   
2. **test_model_forward**: ✓ PASSED (after fixing to 150 channels)
   - Tests model forward pass with dummy data
   - Verifies NaN handling
   
3. **test_training_one_epoch**: ✓ PASSED
   - Fast dev run with small subset
   - Verifies training loop works
   
4. **test_full_training_loop**: ✓ PASSED
   - 2-epoch training with checkpointing
   - Verifies checkpoint saving and loading
   - Verifies CSV logging

## Running the Training

### Basic Usage
```bash
cd benchmark
python train_ictal_interictal.py
```

### With Custom Options
```bash
python train_ictal_interictal.py --batch_size 32 --lr 0.0001 --precision bf16-mixed
```

### Feature Extraction Mode (Frozen Encoder)
```bash
python train_ictal_interictal.py --freeze_encoder
```

## Expected Output

```
Loading dataset...
  Found 106 patient files
  Total windows: 79435
  Interictal (0): 71166 (89.6%)
  Ictal (1): 8269 (10.4%)
✓ Loaded dataset with 79435 windows
✓ Split: Train=55604, Val=11915, Test=11916

Calculating class weights...
✓ Class weights: tensor([0.5588, 4.8043])

Creating model...
  Freeze encoder: False
  Training mode: Fine-tuning (entire model trainable)
  Loading from checkpoint: C:\Users\sirrus\Desktop\ieeg_sz_embedding\checkpoints\...
  Note: Adjusted spatial_transformer_heads from 7 to 8
  ✓ Checkpoint loaded successfully

Starting training...
  Precision: 16-mixed
Epoch 0: 100%|██████| 3475/3475 [45:23<00:00, train_loss=0.652, val_loss=0.589]
...
```

## Files Modified

1. **benchmark/train_ictal_interictal.py**
   - Complete rewrite for NPZ dataset
   - Added checkpoint loading
   - Added mixed precision support
   - Fixed ReduceLROnPlateau verbose parameter

2. **benchmark/dataset_all_windows.py**
   - Added `label_counts` storage
   - Added `get_class_weights()` method

3. **benchmark/test_train_ictal_interictal.py**
   - New comprehensive test suite

## Performance Notes

- **Mixed precision (16-mixed)**: ~2x faster training, ~50% memory reduction
- **Memory-mapped NPZ**: Only loads needed windows, supports datasets > RAM
- **Class weights**: Essential for 89.6/10.4 imbalanced dataset
- **467M parameters**: Requires GPU with ≥8GB VRAM (or CPU with patience)

## Next Steps

1. Run training: `python train_ictal_interictal.py`
2. Monitor with CSV logs in `logs/ictal_interictal/`
3. Checkpoints saved to `benchmark_checkpoints/ictal_interictal/`
4. Evaluate on test set after training completes

## Notes

- Model expects exactly 150 channels (padded in collate function)
- Checkpoint loading uses `strict=False` to allow partial loading
- All data is pre-normalized (z-score per channel) during extraction
- Training is deterministic (fixed random seed)
