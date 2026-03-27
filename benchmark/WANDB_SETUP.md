# Weights & Biases (W&B) Setup Guide

## Overview

Both training scripts now support Weights & Biases for experiment tracking and visualization:
- `train_benchmarks.py`: For benchmark models (ONSET, WVNT, Conformer)
- `train_ictal_interictal.py`: For MultivarWav2Vec2Classifier

## Quick Start

### 1. Install W&B
```bash
pip install wandb
```

### 2. Login to W&B
```bash
wandb login
```

This will open your browser and ask you to:
1. Sign up or log in at https://wandb.ai
2. Copy your API key
3. Paste it in the terminal

Alternatively, set your API key as an environment variable:
```bash
# Windows PowerShell
$env:WANDB_API_KEY="your_api_key_here"

# Windows CMD
set WANDB_API_KEY=your_api_key_here

# Linux/Mac
export WANDB_API_KEY=your_api_key_here
```

### 3. Set Your Entity (Optional)
```bash
# Windows PowerShell
$env:WANDB_ENTITY="your_username_or_team"

# Linux/Mac
export WANDB_ENTITY=your_username_or_team
```

## Usage

### train_benchmarks.py

#### Online Mode (Default)
Logs to W&B in real-time:
```bash
python train_benchmarks.py --model onset --aggregation mean --wandb-mode online
```

#### Offline Mode
Saves logs locally, sync later:
```bash
python train_benchmarks.py --model onset --aggregation mean --wandb-mode offline
```

To sync offline runs later:
```bash
wandb sync wandb_logs/offline-run-*
```

#### Disabled Mode
No W&B logging (only TensorBoard and CSV):
```bash
python train_benchmarks.py --model onset --aggregation mean --wandb-mode disabled
```

### train_ictal_interictal.py

Currently uses CSV logging. To add W&B support, modify the script to include:

```python
from lightning.pytorch.loggers import WandbLogger

# Add to logger configuration
wandb_logger = WandbLogger(
    project='ieeg-seizure-detection',
    name='ictal_interictal',
    save_dir='wandb_logs',
    log_model=True,
    config={
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'freeze_encoder': config.FREEZE_ENCODER
    }
)

# Update trainer
trainer = L.Trainer(
    ...
    logger=[csv_logger, wandb_logger]  # Add wandb_logger
)
```

## W&B Project Configuration

### Default Project Name
```
ieeg-seizure-detection
```

### Run Names
- **Benchmark models**: `{model_name}_{aggregation}` (e.g., `onset_mean`, `wvnt_max`)
- **Ictal/Interictal**: `ictal_interictal`

### Logged Hyperparameters

#### train_benchmarks.py
- `model`: Model name (onset/wvnt/conformer)
- `aggregation`: Channel aggregation method (mean/max)
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay
- `max_epochs`: Maximum epochs
- `freeze_backbone`: Whether backbone is frozen
- `checkpoint_every_n_steps`: Checkpoint frequency

#### Logged Metrics
- **Training**: `train/loss`, `train/acc`, `train/f1`
- **Validation**: `val/loss`, `val/acc`, `val/f1`, `val/precision`, `val/recall`, `val/auroc`
- **Test**: `test/loss`, `test/acc`, `test/f1`, `test/precision`, `test/recall`, `test/auroc`

## Advanced Configuration

### Custom Project Name
Modify in the code:
```python
wandb_logger = WandbLogger(
    project='my-custom-project',  # Change this
    ...
)
```

### Team Collaboration
Set your team as entity:
```python
wandb_logger = WandbLogger(
    project='ieeg-seizure-detection',
    entity='my-team-name',  # Add this
    ...
)
```

### Tags and Groups
Organize runs with tags and groups:
```python
wandb_logger = WandbLogger(
    project='ieeg-seizure-detection',
    name=f'{model_name}_{aggregation}',
    tags=['benchmark', model_name, f'agg_{aggregation}'],  # Add tags
    group=f'{model_name}_experiments',  # Group related runs
    ...
)
```

### Log Additional Data
```python
# In training loop or callbacks
wandb_logger.log_metrics({
    'custom_metric': value,
    'epoch': epoch
})

# Log images, tables, etc.
import wandb
wandb.log({
    'confusion_matrix': wandb.plot.confusion_matrix(
        y_true=labels,
        preds=predictions,
        class_names=['interictal', 'ictal']
    )
})
```

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `WANDB_API_KEY` | Your W&B API key | `abc123...` |
| `WANDB_ENTITY` | Username or team name | `my-lab` |
| `WANDB_PROJECT` | Default project name | `ieeg-seizure-detection` |
| `WANDB_MODE` | Logging mode | `online`, `offline`, `disabled` |
| `WANDB_DIR` | Directory for W&B files | `./wandb_logs` |
| `WANDB_CACHE_DIR` | Cache directory | `./wandb_cache` |
| `WANDB_CONFIG_DIR` | Config directory | `./wandb_config` |

## W&B Dashboard Features

### What You'll See

1. **Overview Dashboard**
   - Loss curves (train/val/test)
   - Accuracy metrics over time
   - System metrics (GPU/CPU usage, memory)

2. **Runs Table**
   - Compare hyperparameters across runs
   - Sort by metrics (e.g., best val_auroc)
   - Filter runs by tags or config

3. **Charts**
   - Customizable plots
   - Compare multiple runs
   - Export high-quality figures

4. **Model Artifacts**
   - Saved checkpoints
   - Model architecture
   - Download/restore models

5. **System Metrics**
   - GPU utilization
   - Memory usage
   - Training speed

### Useful Commands

```bash
# List recent runs
wandb sync --show

# Sync specific offline run
wandb sync wandb_logs/offline-run-20260212_123456-abc123

# View run in browser
wandb online  # Set mode to online

# Clean up old runs
wandb artifact cache cleanup 10GB  # Keep only 10GB of artifacts
```

## Troubleshooting

### Issue: "Not logged in"
**Solution:**
```bash
wandb login
# or
export WANDB_API_KEY=your_key
```

### Issue: "Connection timeout"
**Solution:** Use offline mode:
```bash
python train_benchmarks.py --model onset --wandb-mode offline
```

### Issue: "Permission denied" on team project
**Solution:** Set entity:
```bash
export WANDB_ENTITY=your_team_name
```

### Issue: Too many runs cluttering dashboard
**Solution:** Use groups and tags:
```python
wandb_logger = WandbLogger(
    ...,
    group='experiment_1',
    tags=['baseline', 'onset']
)
```

### Issue: Large model files uploading slowly
**Solution:** Disable model logging:
```python
wandb_logger = WandbLogger(
    ...,
    log_model=False  # Don't upload checkpoints
)
```

## Example Workflows

### Hyperparameter Sweep
```bash
# Test different aggregations
for agg in mean max; do
    python train_benchmarks.py --model onset --aggregation $agg --wandb-mode online
done
```

### Debugging (No W&B)
```bash
python train_benchmarks.py --model onset --wandb-mode disabled --max-epochs 1
```

### Production Training (Offline)
```bash
# Train offline
python train_benchmarks.py --model onset --wandb-mode offline --max-epochs 100

# Sync later
wandb sync wandb_logs/offline-*
```

## Resources

- **W&B Docs**: https://docs.wandb.ai/
- **Python Library**: https://docs.wandb.ai/ref/python
- **PyTorch Lightning Integration**: https://docs.wandb.ai/guides/integrations/lightning
- **Dashboard**: https://wandb.ai/home

## Summary

✅ **Install**: `pip install wandb`  
✅ **Login**: `wandb login`  
✅ **Train**: `python train_benchmarks.py --model onset --wandb-mode online`  
✅ **View**: Check your dashboard at https://wandb.ai  
