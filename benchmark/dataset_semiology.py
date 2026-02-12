"""
Dataset for semiology classification (multi-class).

Loads 10-second ictal windows and extracts semiology labels for supervised training.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter

from config_benchmark import BenchmarkConfig


class SemiologyDataset(Dataset):
    """
    Dataset for seizure semiology classification.
    
    Loads ictal data and assigns semiology labels from metadata.
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
        
        # Load data
        print(f"Loading semiology dataset ({split} split)...")
        self._load_data()
        
        # Create semiology labels
        self._create_labels()
        
        # Split data
        self._split_data()
        
        print(f"✓ Loaded {len(self)} samples for {split} split")
        print(f"  - Number of classes: {self.num_classes}")
        print(f"  - Class distribution: {dict(Counter(self.labels))}")
    
    def _load_data(self):
        """Load preprocessed ictal data."""
        # Load pickle file
        with open(self.config.ICTAL_DATA_FILE, 'rb') as f:
            signals, coords, regs, ch_names, indices = pickle.load(f)
        
        # Convert to tensors
        self.signals = [torch.from_numpy(sig).float() for sig in signals]
        self.coords = [torch.from_numpy(coord).float() for coord in coords]
        self.regs = [torch.from_numpy(reg).long() + 1 for reg in regs]  # +1 offset
        self.ch_names = ch_names
        self.indices = indices
        
        # Load metadata
        self._load_metadata()
        
        # Filter metadata to match loaded indices
        self.metadata = self.sz_table.loc[indices].reset_index(drop=True)
        
        print(f"  - Loaded {len(self.signals)} seizures")
        print(f"  - Metadata shape: {self.metadata.shape}")
    
    def _load_metadata(self):
        """Load seizure metadata and create patient mappings."""
        # Load seizure times
        sz_times = pd.read_excel(
            self.config.SEIZURE_METADATA_FILE,
            sheet_name="AllSeizureTimes",
            index_col=0,
        )
        
        # Load RID-HUP mapping
        rid_hup_table = pd.read_csv(self.config.RID_HUP_TABLE_FILE, index_col=0)
        rid_hup_table.dropna(inplace=True, subset=["hupsubjno"])
        
        # Extract first 3 characters and convert to int
        rid_hup_table['hupsubjno'] = rid_hup_table['hupsubjno'].str[:3].astype(int)
        
        rid_hup_table.index = [f"sub-RID{x:04d}" for x in rid_hup_table.index]
        rid_hup_table['hupsubjno'] = [f"HUP{x:03d}" for x in rid_hup_table['hupsubjno']]
        
        # Create mappings
        hup_to_rid = {v: k for k, v in rid_hup_table['hupsubjno'].to_dict().items()}
        
        # Clean seizure times
        sz_times.dropna(inplace=True, subset=["IEEGname"])
        sz_times = sz_times[sz_times["IEEGname"] != "HUP203_phaseII"]
        
        # Create sz_table with RID indices
        sz_table = sz_times.copy()
        sz_table.index = sz_table.index.map(hup_to_rid)
        sz_table.reset_index(inplace=True)
        
        self.sz_table = sz_table
        self.hup_to_rid = hup_to_rid
    
    def _create_labels(self):
        """Create semiology labels from metadata."""
        # Extract semiology column
        semiologies = self.metadata['Semiology'].copy()
        
        # Handle NaN values
        semiologies = semiologies.fillna('Unknown')
        
        # Clean up semiology strings
        semiologies = semiologies.str.strip()
        
        # Count occurrences
        semiology_counts = Counter(semiologies)
        print(f"  - Found {len(semiology_counts)} unique semiology types")
        
        # Filter out rare classes
        min_samples = self.config.MIN_SAMPLES_PER_CLASS
        valid_semiologies = {sem for sem, count in semiology_counts.items() 
                            if count >= min_samples}
        
        # Replace rare semiologies with 'Other'
        semiologies = semiologies.apply(
            lambda x: x if x in valid_semiologies else 'Other'
        )
        
        # Create label encoding
        unique_semiologies = sorted(semiologies.unique())
        self.semiology_to_idx = {sem: idx for idx, sem in enumerate(unique_semiologies)}
        self.idx_to_semiology = {idx: sem for sem, idx in self.semiology_to_idx.items()}
        
        # Convert to indices
        self.all_labels = semiologies.map(self.semiology_to_idx).values
        
        # Store patient IDs for stratified splitting
        self.all_patients = self.metadata['Patient'].values
        
        self.num_classes = len(self.semiology_to_idx)
        
        print(f"  - After filtering (min {min_samples} samples): {self.num_classes} classes")
        print(f"  - Classes: {list(self.semiology_to_idx.keys())}")
    
    def _split_data(self):
        """Split data into train/val/test sets (patient-stratified)."""
        # Get unique patients
        unique_patients = np.unique(self.all_patients)
        n_patients = len(unique_patients)
        
        # Create patient-level labels (most common semiology per patient)
        patient_labels = {}
        for patient in unique_patients:
            mask = self.all_patients == patient
            patient_semiologies = self.all_labels[mask]
            # Use most common semiology for this patient
            patient_labels[patient] = Counter(patient_semiologies).most_common(1)[0][0]
        
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
            label: Semiology class index
            sample_idx: Original sample index in full dataset
        """
        # Map to original index
        original_idx = self.split_indices[idx]
        
        # Get data
        signals = self.signals[original_idx]
        regs = self.regs[original_idx]
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
        if not self.config.USE_CLASS_WEIGHTS:
            return None
        
        # Count samples per class
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        
        # Compute inverse frequency weights
        weights = torch.zeros(self.num_classes)
        for class_idx in range(self.num_classes):
            count = class_counts.get(class_idx, 1)  # Avoid division by zero
            weights[class_idx] = total_samples / (self.num_classes * count)
        
        return weights
    
    def get_label_names(self):
        """Get the list of semiology names."""
        return [self.idx_to_semiology[i] for i in range(self.num_classes)]


if __name__ == "__main__":
    # Test the dataset
    print("Testing SemiologyDataset...")
    
    # Create datasets
    train_dataset = SemiologyDataset(split='train')
    val_dataset = SemiologyDataset(split='val')
    test_dataset = SemiologyDataset(split='test')
    
    print(f"\n✓ Created datasets:")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Val: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")
    
    # Test data loading
    signals, regs, label, idx = train_dataset[0]
    print(f"\n✓ Sample data:")
    print(f"  - Signals shape: {signals.shape}")
    print(f"  - Regs shape: {regs.shape}")
    print(f"  - Label: {label} ({train_dataset.idx_to_semiology[label]})")
    
    # Test class weights
    if train_dataset.config.USE_CLASS_WEIGHTS:
        weights = train_dataset.get_class_weights()
        print(f"\n✓ Class weights shape: {weights.shape}")
