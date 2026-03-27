# Ictal/Interictal Training Script - Setup Complete

## Summary of Changes

All paths have been fixed and `freeze_encoder` has been set to `False` by default for fine-tuning the entire model.

## Files Modified

### 1. `benchmark/train_ictal_interictal.py`

**Path Fixes:**
- Added benchmark directory to Python path
- Added code directory to Python path with existence check
- Uses absolute paths with proper path joining

**Freeze Encoder Changes:**
- Default is now `False` (fine-tune entire model)
- Added helpful status messages showing training mode
- User can still pass `--freeze_encoder` flag to enable feature extraction mode

**Key Changes:**
```python
# Before
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

# After
benchmark_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, benchmark_dir)
code_dir = os.path.join(os.path.dirname(benchmark_dir), 'code')
if os.path.exists(code_dir):
    sys.path.insert(0, code_dir)
```

**Training Mode Display:**
```python
print(f"  Freeze encoder: {config.FREEZE_ENCODER}")
print(f"  Training mode: {'Feature extraction (frozen encoder)' if config.FREEZE_ENCODER else 'Fine-tuning (entire model trainable)'}")
```

---

### 2. `benchmark/model_classifier.py`

**Path Fixes:**
- Fixed import paths to avoid triggering problematic `config.py`
- Reimplemented helper classes (Transpose, PositionalEncoding, Transformer)
- Added fallback MODEL_CONFIG
- Made imports safe and self-contained

**Model Compatibility Fixes:**
- Added automatic adjustment of transformer heads to ensure compatibility
- Ensures `hidden_dim % num_heads == 0` for all transformers
- Adjusts spatial_transformer_heads (e.g., 7 → 8 for 640 hidden dim)

**Test Results:**
```
✓ Forward pass successful
  Input shape: torch.Size([2, 150, 2560])
  Output shape: torch.Size([2, 5])

✓ Embedding extraction successful
  Embeddings shape: torch.Size([2, 4800])
```

---

## Configuration

**Default Settings (config_benchmark.py):**
- `FREEZE_ENCODER = False` ✓ (Fine-tune entire model)
- All paths use local directories
- No network mount dependencies

---

## Usage

### 1. Train with Default Settings (Fine-tune Entire Model)

```bash
cd c:\Users\sirrus\Desktop\ieeg_sz_embedding\benchmark
..\.venv\Scripts\python.exe train_ictal_interictal.py
```

**Expected Output:**
```
Creating model...
  Freeze encoder: False
  Training mode: Fine-tuning (entire model trainable)
```

### 2. Train with Frozen Encoder (Feature Extraction)

```bash
python train_ictal_interictal.py --freeze_encoder
```

**Expected Output:**
```
Creating model...
  Freeze encoder: True
  Training mode: Feature extraction (frozen encoder)
✓ Encoder frozen (feature extraction mode)
```

### 3. Other Options

```bash
# Custom batch size
python train_ictal_interictal.py --batch_size 16

# Custom learning rate
python train_ictal_interictal.py --lr 0.0001

# Load pretrained checkpoint
python train_ictal_interictal.py --pretrained_ckpt path/to/checkpoint.ckpt

# Combine options
python train_ictal_interictal.py --batch_size 32 --lr 0.001 --pretrained_ckpt checkpoint.ckpt
```

### 4. View Help

```bash
python train_ictal_interictal.py --help
```

---

## Architecture

### MultivarWav2Vec2Classifier

**Components:**
1. **Feature Encoder**: 1D convolutions with adaptive pooling
2. **Positional Encoding**: Spatial (region) and temporal (frame) embeddings
3. **Spatial Transformer**: Processes across channels within each frame
4. **Temporal Transformer**: Processes across frames with CLS token
5. **Classification Head**: 2-layer MLP with dropout

**Input:**
- Signals: `(batch, channels, timesteps)` - e.g., `(2, 150, 2560)`
- Regions: `(batch, channels)` - e.g., `(2, 150)`

**Output:**
- Logits: `(batch, num_classes)` - e.g., `(2, 2)` for binary classification

**Embeddings:**
- CLS token: `(batch, embedding_dim)` - e.g., `(2, 4800)`

---

## Training Details

**Default Hyperparameters:**
- Learning rate: From config (typically 1e-3)
- Optimizer: Adam with weight decay
- Scheduler: ReduceLROnPlateau
- Batch size: From config (typically 32)
- Max epochs: From config (typically 100)
- Early stopping: Patience from config

**Metrics Logged:**
- Accuracy
- F1 Score
- Precision
- Recall
- AUROC (test only)
- Confusion Matrix (validation & test)

**Callbacks:**
- ModelCheckpoint: Saves top 3 models by validation loss
- EarlyStopping: Stops training if no improvement
- CSVLogger: Logs all metrics to CSV

---

## Checkpoints

**Location:**
```
benchmark_checkpoints/ictal_interictal/
```

**Filename Format:**
```
ictal_interictal-epoch={epoch:02d}-val_loss={val_loss:.4f}-val_acc={val_acc:.4f}.ckpt
```

---

## Logs

**Location:**
```
logs/ictal_interictal/
```

**Files:**
- `version_X/metrics.csv` - All training metrics
- `version_X/hparams.yaml` - Hyperparameters used

---

## Verification

All imports and model functionality have been tested:

✅ `train_ictal_interictal.py` imports successfully
✅ `model_classifier.py` imports successfully
✅ MODEL_CONFIG loaded correctly
✅ MultivarWav2Vec2Classifier instantiates
✅ Forward pass works (150 channels, 2560 timesteps)
✅ Embedding extraction works (4800-dim)
✅ freeze_encoder defaults to False
✅ Help text displays correctly

---

## Next Steps

1. **Ensure Data is Ready:**
   - Extract windows using `extract_all_windows.py`
   - Verify NPZ files exist in `data/all_windows_per_patient/`

2. **Create Dataset Splits:**
   - Verify `dataset_ictal_interictal.py` can load data
   - Check train/val/test splits

3. **Start Training:**
   ```bash
   python train_ictal_interictal.py
   ```

4. **Monitor Training:**
   - Watch terminal output for metrics
   - Check CSV logs in `logs/ictal_interictal/`

5. **Evaluate:**
   - Best model automatically selected
   - Test metrics printed at end
   - Confusion matrix displayed

---

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure:
- Virtual environment is activated
- Working directory is correct
- All dependencies are installed

### Model Dimension Errors

The script automatically adjusts transformer heads if needed:
```
Note: Adjusted spatial_transformer_heads from 7 to 8
```

This is expected and ensures compatibility.

### Memory Issues

If you run out of memory:
```bash
python train_ictal_interictal.py --batch_size 8
```

Or adjust `NUM_WORKERS` in config to 0 for Windows.

---

## Testing

To test the model without training:

```python
import torch
from model_classifier import MultivarWav2Vec2Classifier

# Create model
model = MultivarWav2Vec2Classifier(num_classes=2, freeze_encoder=False)

# Test input
x = torch.randn(2, 94, 2560)  # batch=2, channels=94, timesteps=2560
regs = torch.randint(0, 41, (2, 94))  # region labels

# Forward pass
logits = model(x, regs)
print(f"Output shape: {logits.shape}")  # Should be (2, 2)
```

---

## Summary

🎉 **Setup Complete!**

- ✅ All paths fixed
- ✅ Imports working
- ✅ Model tested
- ✅ Default: Fine-tune entire model (`freeze_encoder=False`)
- ✅ Ready for training

**Key Command:**
```bash
cd c:\Users\sirrus\Desktop\ieeg_sz_embedding\benchmark
..\.venv\Scripts\python.exe train_ictal_interictal.py
```
