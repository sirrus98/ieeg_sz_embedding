"""
Dataset for ictal vs interictal classification (binary).

Loads both ictal and interictal 10-second windows for supervised binary classification.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter

from config_benchmark import BenchmarkConfig


class IctalInterictalDataset(Dataset):
    """
    Dataset for ictal vs interictal binary classification.
    
    Combines ictal (label=1) and interictal (label=0) data.
    """
    
    def __init__(self, config=BenchmarkConfig, split='train', random_seed=42):
        """
        Initialize the dataset.
        
        Args:
            config: Configuration class with paths and parameters
            split: 'train', 'val', or 'test'
            random_seed: Random seed for reproducible splits
        """
        self.config = config
        self.split = split
        self.random_seed = random_seed
        self.num_classes = 2
        
        # Load data
        print(f"Loading ictal/interictal dataset ({split} split)...")
        self._load_ictal_data()
        self._load_interictal_data()
        
        # Combine datasets
        self._combine_data()
        
        # Split data (patient-stratified)
        self._split_data()
        
        print(f"✓ Loaded {len(self)} samples for {split} split")
        print(f"  - Class distribution: {dict(Counter(self.labels))}")
    
    def _load_ictal_data(self):
        """Load preprocessed ictal (seizure) data."""
        print("  Loading ictal data...")
        with open(self.config.ICTAL_DATA_FILE, 'rb') as f:
            signals, coords, regs, ch_names, indices = pickle.load(f)
        
        # Convert to tensors
        self.ictal_signals = [torch.from_numpy(sig).float() for sig in signals]
        self.ictal_coords = [torch.from_numpy(coord).float() for coord in coords]
        self.ictal_regs = [torch.from_numpy(reg).long() + 1 for reg in regs]
        self.ictal_ch_names = ch_names
        
        # For ictal data, we need to map indices to patient IDs
        # Load metadata to get patient info
        import pandas as pd
        
        sz_times = pd.read_excel(
            self.config.SEIZURE_METADATA_FILE,
            sheet_name="AllSeizureTimes",
            index_col=0,
        )
        
        rid_hup_table = pd.read_csv(self.config.RID_HUP_TABLE_FILE, index_col=0)
        rid_hup_table.dropna(inplace=True, subset=["hupsubjno"])
        
        for ind, row in rid_hup_table.iterrows():
            rid_hup_table.loc[ind, "hupsubjno"] = int(row["hupsubjno"][:3])
        
        rid_hup_table.index = [f"sub-RID{x:04d}" for x in rid_hup_table.index]
        rid_hup_table['hupsubjno'] = [f"HUP{x:03d}" for x in rid_hup_table['hupsubjno']]
        
        hup_to_rid = {v: k for k, v in rid_hup_table['hupsubjno'].to_dict().items()}
        
        sz_times.dropna(inplace=True, subset=["IEEGname"])
        sz_times = sz_times[sz_times["IEEGname"] != "HUP203_phaseII"]
        
        sz_table = sz_times.copy()
        sz_table.index = sz_table.index.map(hup_to_rid)
        sz_table.reset_index(inplace=True)
        
        # Get patient IDs for ictal samples
        metadata = sz_table.loc[indices].reset_index(drop=True)
        self.ictal_patients = metadata['Patient'].values
        
        print(f"    - {len(self.ictal_signals)} ictal samples from {len(set(self.ictal_patients))} patients")
    
    def _load_interictal_data(self):
        """Load preprocessed interictal (non-seizure) data."""
        print("  Loading interictal data...")
        
        if not os.path.exists(self.config.INTERICTAL_DATA_FILE):
            raise FileNotFoundError(
                f"Interictal data not found at: {self.config.INTERICTAL_DATA_FILE}\n"
                f"Please run extract_interictal_data.py first!"
            )
        
        with open(self.config.INTERICTAL_DATA_FILE, 'rb') as f:
            signals, coords, regs, ch_names, patient_indices = pickle.load(f)
        
        # Convert to tensors
        self.interictal_signals = [torch.from_numpy(sig).float() for sig in signals]
        self.interictal_coords = [torch.from_numpy(coord).float() for coord in coords]
        self.interictal_regs = [torch.from_numpy(reg).long() + 1 for reg in regs]
        self.interictal_ch_names = ch_names
        self.interictal_patients = np.array(patient_indices)
        
        print(f"    - {len(self.interictal_signals)} interictal samples from {len(set(self.interictal_patients))} patients")
    
    def _combine_data(self):
        """Combine ictal and interictal data."""
        # Combine signals
        self.all_signals = self.ictal_signals + self.interictal_signals
        self.all_coords = self.ictal_coords + self.interictal_coords
        self.all_regs = self.ictal_regs + self.interictal_regs
        self.all_ch_names = self.ictal_ch_names + self.interictal_ch_names
        
        # Create labels: 1 for ictal, 0 for interictal
        self.all_labels = np.concatenate([
            np.ones(len(self.ictal_signals), dtype=np.int64),
            np.zeros(len(self.interictal_signals), dtype=np.int64)
        ])
        
        # Combine patient IDs
        self.all_patients = np.concatenate([
            self.ictal_patients,
            self.interictal_patients
        ])
        
        print(f"  Combined dataset:")
        print(f"    - Total samples: {len(self.all_signals)}")
        print(f"    - Ictal: {(self.all_labels == 1).sum()}")
        print(f"    - Interictal: {(self.all_labels == 0).sum()}")
        print(f"    - Total patients: {len(set(self.all_patients))}")
    
    def _split_data(self):
        """Split data into train/val/test sets (patient-stratified)."""
        # Get unique patients
        unique_patients = np.unique(self.all_patients)
        n_patients = len(unique_patients)
        
        # Create patient-level labels (whether patient has ictal samples)
        patient_labels = {}
        for patient in unique_patients:
            mask = self.all_patients == patient
            patient_semiologies = self.all_labels[mask]
            # Use majority class for stratification
            patient_labels[patient] = 1 if patient_semiologies.sum() > len(patient_semiologies) / 2 else 0
        
        patient_label_array = np.array([patient_labels[p] for p in unique_patients])
        
        # Split patients
        train_patients, test_val_patients, _, test_val_labels = train_test_split(
            unique_patients, 
            patient_label_array,
            test_size=(self.config.VAL_RATIO + self.config.TEST_RATIO),
            stratify=patient_label_array,
            random_state=self.random_seed
        )
        
        val_patients, test_patients = train_test_split(
            test_val_patients,
            test_size=self.config.TEST_RATIO / (self.config.VAL_RATIO + self.config.TEST_RATIO),
            stratify=test_val_labels,
            random_state=self.random_seed
        )
        
        # Create sample indices for each split
        if self.split == 'train':
            split_patients = set(train_patients)
        elif self.split == 'val':
            split_patients = set(val_patients)
        else:  # test
            split_patients = set(test_patients)
        
        # Filter samples
        split_mask = np.array([p in split_patients for p in self.all_patients])
        split_indices = np.where(split_mask)[0]
        
        self.split_indices = split_indices
        self.labels = self.all_labels[split_indices]
        
        print(f"  - Split: {len(train_patients)} train / {len(val_patients)} val / {len(test_patients)} test patients")
    
    def __len__(self):
        """Return the number of samples in this split."""
        return len(self.split_indices)
    
    def __getitem__(self, idx):
        """
        Get a sample.
        
        Returns:
            signals: Preprocessed iEEG signals (channels, timesteps)
            regs: Region labels (channels,)
            label: Binary class (0=interictal, 1=ictal)
            sample_idx: Original sample index in full dataset
        """
        # Map to original index
        original_idx = self.split_indices[idx]
        
        # Get data
        signals = self.all_signals[original_idx]
        regs = self.all_regs[original_idx]
        label = self.labels[idx]
        
        # Pad signals and regs to channel_buffer_size
        n_channels = signals.shape[0]
        if n_channels < self.config.CHANNEL_BUFFER_SIZE:
            # Pad signals
            pad_size = self.config.CHANNEL_BUFFER_SIZE - n_channels
            signals = torch.nn.functional.pad(signals, (0, 0, 0, pad_size))
            # Pad regs with 0 (no region)
            regs = torch.nn.functional.pad(regs, (0, pad_size))
        
        return signals, regs, label, original_idx
    
    def get_class_weights(self):
        """
        Compute class weights for handling imbalance.
        
        Returns:
            class_weights: Tensor of weights for each class
        """
        # Count samples per class
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        
        # Compute inverse frequency weights
        weights = torch.zeros(self.num_classes)
        for class_idx in range(self.num_classes):
            count = class_counts.get(class_idx, 1)
            weights[class_idx] = total_samples / (self.num_classes * count)
        
        return weights
    
    def get_label_names(self):
        """Get the list of class names."""
        return ['Interictal', 'Ictal']


if __name__ == "__main__":
    # Test the dataset
    print("Testing IctalInterictalDataset...")
    
    # Create datasets
    train_dataset = IctalInterictalDataset(split='train')
    val_dataset = IctalInterictalDataset(split='val')
    test_dataset = IctalInterictalDataset(split='test')
    
    print(f"\n✓ Created datasets:")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Val: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")
    
    # Test data loading
    signals, regs, label, idx = train_dataset[0]
    print(f"\n✓ Sample data:")
    print(f"  - Signals shape: {signals.shape}")
    print(f"  - Regs shape: {regs.shape}")
    print(f"  - Label: {label} ({train_dataset.get_label_names()[label]})")
    
    # Test class weights
    weights = train_dataset.get_class_weights()
    print(f"\n✓ Class weights: {weights}")
