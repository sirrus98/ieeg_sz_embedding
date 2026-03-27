"""Regenerate the complete debug_dataset_cnn.ipynb notebook"""
import json
import os

# Complete notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    }

def add_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code]
    }

# Build cells
cells = [
    # Cell 0: Title
    add_markdown_cell("""# Debug iEEG Seizure Dataset with 1D CNN Classifier

This notebook debugs the globally-normalized iEEG seizure dataset by:
1. Visualizing signals for ictal vs interictal windows
2. Computing and comparing statistics (range, std, line length) by label
3. Training a simple 1D CNN classifier on single-channel data

**Data Source**: `data/all_windows_per_patient_global_norm/`
- Global normalization: mean=2.88e-09, std=0.000239
- Selected patients: sub-RID0106, sub-RID0013, sub-RID0020
- Window shape: (n_channels, 2560) at 256 Hz"""),
    
    # Cell 1: Section 1
    add_markdown_cell("## 1. Setup and Data Loading"),
    
    # Cell 2: Imports
    add_code_cell("""# Imports
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("✓ Imports complete")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")"""),
    
    # Cell 3: Configuration
    add_code_cell("""# Add code/benchmark directories to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root / 'benchmark'))
sys.path.insert(0, str(project_root / 'code'))

from config_benchmark import BenchmarkConfig

# Configuration
config = BenchmarkConfig()
data_dir = Path(config.OUTPUT_DATA_DIR) / "all_windows_per_patient_global_norm"

# Selected patients for analysis
selected_patients = ['sub-RID0106', 'sub-RID0013', 'sub-RID0020']

# Load global normalization stats
with open(Path(config.OUTPUT_DATA_DIR) / 'global_normalization_stats.json', 'r') as f:
    global_stats = json.load(f)

print("✓ Configuration loaded")
print(f"Data directory: {data_dir}")
print(f"\\nGlobal normalization stats:")
print(f"  Mean: {global_stats['global_mean']:.2e}")
print(f"  Std: {global_stats['global_std']:.6f}")
print(f"  Files sampled: {global_stats['n_files_sampled']}")"""),
    
    # Cell 4: Load data
    add_code_cell("""# Load data for selected patients
print("Loading patient data...")

all_signals = []
all_coords = []
all_regs = []
all_labels = []
all_patient_ids = []
all_n_channels = []

# First pass: find maximum channel count
max_channels = 0
patient_data_list = []

for patient_id in selected_patients:
    file_path = data_dir / f"{patient_id}.npz"
    
    if not file_path.exists():
        print(f"  ✗ {patient_id}: File not found")
        continue
    
    data = np.load(file_path, allow_pickle=True)
    
    signals = data['signals'].astype(np.float32)
    coords = data['coords']
    regs = data['regs']
    labels = data['labels']
    n_channels = data['n_channels'] if 'n_channels' in data else np.full(len(signals), signals.shape[1])
    
    n_windows = len(signals)
    n_ictal = np.sum(labels == 1)
    n_interictal = np.sum(labels == 0)
    
    # Track max channels across all patients
    max_channels = max(max_channels, signals.shape[1])
    
    print(f"  ✓ {patient_id}: {n_windows} windows, {signals.shape[1]} channels ({n_ictal} ictal, {n_interictal} interictal)")
    
    patient_data_list.append({
        'patient_id': patient_id,
        'signals': signals,
        'coords': coords,
        'regs': regs,
        'labels': labels,
        'n_channels': n_channels
    })

print(f"\\n  Max channels across patients: {max_channels}")

# Second pass: pad all to max_channels and concatenate
for patient_data in patient_data_list:
    signals = patient_data['signals']
    coords = patient_data['coords']
    regs = patient_data['regs']
    labels = patient_data['labels']
    n_channels = patient_data['n_channels']
    patient_id = patient_data['patient_id']
    
    n_windows, curr_channels, timesteps = signals.shape
    
    # Pad if necessary
    if curr_channels < max_channels:
        # Pad signals with zeros
        padded_signals = np.zeros((n_windows, max_channels, timesteps), dtype=np.float32)
        padded_signals[:, :curr_channels, :] = signals
        
        # Pad coords with zeros
        padded_coords = np.zeros((n_windows, max_channels, 3), dtype=coords.dtype)
        padded_coords[:, :curr_channels, :] = coords
        
        # Pad regs with zeros
        padded_regs = np.zeros((n_windows, max_channels), dtype=regs.dtype)
        padded_regs[:, :curr_channels] = regs
        
        signals = padded_signals
        coords = padded_coords
        regs = padded_regs
    
    all_signals.append(signals)
    all_coords.append(coords)
    all_regs.append(regs)
    all_labels.append(labels)
    all_patient_ids.extend([patient_id] * n_windows)
    all_n_channels.append(n_channels)

# Concatenate all data
all_signals = np.concatenate(all_signals, axis=0)
all_coords = np.concatenate(all_coords, axis=0)
all_regs = np.concatenate(all_regs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_n_channels = np.concatenate(all_n_channels, axis=0)

print(f"\\n✓ Total dataset:")
print(f"  Windows: {len(all_labels)}")
print(f"  Shape: {all_signals.shape} (padded to {max_channels} channels)")
print(f"  Ictal (1): {np.sum(all_labels == 1)} ({100*np.sum(all_labels == 1)/len(all_labels):.1f}%)")
print(f"  Interictal (0): {np.sum(all_labels == 0)} ({100*np.sum(all_labels == 0)/len(all_labels):.1f}%)")
print(f"  Signal range: [{np.nanmin(all_signals):.4f}, {np.nanmax(all_signals):.4f}]")
print(f"  Signal mean: {np.nanmean(all_signals):.6f}")
print(f"  Signal std: {np.nanstd(all_signals):.6f}")"""),
]

# Save notebook
notebook["cells"] = cells
output_path = "debug_dataset_cnn_part1.json"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Generated first part of notebook: {output_path}")
print(f"  Total cells: {len(cells)}")
