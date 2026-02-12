"""
PyTorch Dataset for lazy-loading windows from HDF5 file.

This dataset loads windows on-the-fly from an HDF5 file, avoiding loading
the entire dataset into memory at once.

Data is already preprocessed and normalized during extraction (following make_dataset.py):
- Per-channel z-score normalization: (x - nanmean(x)) / nanstd(x)
- No additional normalization needed in dataset
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json


class HDF5WindowDataset(Dataset):
    """
    PyTorch Dataset that lazy-loads windows from per-patient HDF5 files.
    
    Each window contains:
    - signals: (n_channels, n_timesteps) EEG data
    - coords: (n_channels, 3) MNI coordinates
    - regs: (n_channels,) brain region labels
    - label: 0 (interictal) or 1 (ictal)
    - metadata: patient_id, channel_names, window_info
    """
    
    def __init__(self, h5_dir_path, transform=None, return_metadata=False):
        """
        Initialize the dataset from a directory of per-patient HDF5 files.
        
        Note: Data is already normalized during extraction (per-channel z-score).
        Additional transforms should not include normalization.
        
        Args:
            h5_dir_path: Path to directory containing per-patient HDF5 files
            transform: Optional transform to apply to signals (e.g., augmentation)
            return_metadata: If True, return metadata dict with each sample
        """
        self.h5_dir_path = h5_dir_path
        self.transform = transform
        self.return_metadata = return_metadata
        
        # Find all patient HDF5 files
        from glob import glob
        self.patient_files = sorted(glob(os.path.join(h5_dir_path, "*.h5")))
        
        if len(self.patient_files) == 0:
            raise ValueError(f"No HDF5 files found in {h5_dir_path}")
        
        print(f"Loading HDF5 dataset from directory: {h5_dir_path}")
        print(f"  Found {len(self.patient_files)} patient files")
        
        # Build index: map global index to (patient_file_idx, window_idx_in_file)
        self.index_map = []  # List of (file_idx, window_idx) tuples
        self.patient_ids = []
        
        total_interictal = 0
        total_ictal = 0
        
        for file_idx, patient_file in enumerate(self.patient_files):
            with h5py.File(patient_file, 'r') as f:
                n_windows = f['signals'].shape[0]
                patient_id = f['patient_id'][()].decode('utf-8')
                
                # Add entries to index map
                for window_idx in range(n_windows):
                    self.index_map.append((file_idx, window_idx))
                    self.patient_ids.append(patient_id)
                
                # Count labels
                labels = f['labels'][:]
                total_interictal += np.sum(labels == 0)
                total_ictal += np.sum(labels == 1)
        
        self.length = len(self.index_map)
        
        # Get dimensions from first file
        with h5py.File(self.patient_files[0], 'r') as f:
            self.n_channels = f['signals'].shape[1]
            self.n_timesteps = f['signals'].shape[2]
        
        print(f"  Total windows: {self.length}")
        print(f"  Window shape: ({self.n_channels}, {self.n_timesteps})")
        print(f"  Interictal (0): {total_interictal} ({100*total_interictal/self.length:.1f}%)")
        print(f"  Ictal (1): {total_ictal} ({100*total_ictal/self.length:.1f}%)")
        
        # Cache for file handles (one per worker)
        self._file_handles = {}
    
    def _get_h5_file(self, file_idx):
        """
        Get or create HDF5 file handle for a specific patient file.
        Each worker needs its own file handles for thread safety.
        
        Args:
            file_idx: Index of patient file in self.patient_files
        
        Returns:
            h5py.File handle
        """
        if file_idx not in self._file_handles:
            self._file_handles[file_idx] = h5py.File(self.patient_files[file_idx], 'r')
        return self._file_handles[file_idx]
    
    def __len__(self):
        """Return the number of windows in the dataset."""
        return self.length
    
    def __getitem__(self, idx):
        """
        Get a single window by index.
        
        Args:
            idx: Index of the window to retrieve
        
        Returns:
            If return_metadata=False:
                (signals, coords, regs, label)
            If return_metadata=True:
                (signals, coords, regs, label, metadata_dict)
        """
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range [0, {self.length})")
        
        # Get patient file and window index within that file
        file_idx, window_idx = self.index_map[idx]
        
        # Get file handle for this patient
        f = self._get_h5_file(file_idx)
        
        # Load data for this index (only loads this one window)
        # Data is stored as float16 to save space, convert to float32 for PyTorch
        signals = torch.from_numpy(f['signals'][window_idx].astype(np.float32))
        coords = torch.from_numpy(f['coords'][window_idx]).float()
        regs = torch.from_numpy(f['regs'][window_idx]).long()
        label = int(f['labels'][window_idx])
        
        # Handle padding: get actual channel count if available
        if 'n_channels' in f:
            n_ch = int(f['n_channels'][window_idx])
            # Only keep the actual channels (remove padding)
            signals = signals[:n_ch]
            coords = coords[:n_ch]
            regs = regs[:n_ch]
        
        # Apply transform if provided
        if self.transform is not None:
            signals = self.transform(signals)
        
        if self.return_metadata:
            # Load metadata
            patient_id = self.patient_ids[idx]
            ch_names = json.loads(f['ch_names'][window_idx].decode('utf-8'))
            window_info = json.loads(f['window_info'][window_idx].decode('utf-8'))
            
            metadata = {
                'patient_id': patient_id,
                'ch_names': ch_names,
                'window_info': window_info,
                'index': idx
            }
            
            return signals, coords, regs, label, metadata
        else:
            return signals, coords, regs, label
    
    def get_label(self, idx):
        """Get just the label for a given index (fast)."""
        file_idx, window_idx = self.index_map[idx]
        f = self._get_h5_file(file_idx)
        return int(f['labels'][window_idx])
    
    def get_patient_id(self, idx):
        """Get just the patient ID for a given index."""
        return self.patient_ids[idx]
    
    def get_labels(self):
        """Get all labels (loads from all patient files)."""
        labels = []
        for file_idx, patient_file in enumerate(self.patient_files):
            with h5py.File(patient_file, 'r') as f:
                labels.extend(f['labels'][:].tolist())
        return np.array(labels)
    
    def get_patient_ids(self):
        """Get all patient IDs (already loaded in __init__)."""
        return self.patient_ids
    
    def __del__(self):
        """Close all HDF5 file handles when dataset is destroyed."""
        for f in self._file_handles.values():
            if f is not None:
                f.close()


def create_dataloader(h5_dir_path, batch_size=32, shuffle=True, num_workers=4, 
                      return_metadata=False, pin_memory=True, collate_fn=None):
    """
    Create a PyTorch DataLoader for the HDF5 dataset.
    
    Args:
        h5_dir_path: Path to directory containing per-patient HDF5 files
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        return_metadata: Whether to return metadata with each sample
        pin_memory: Pin memory for faster GPU transfer
        collate_fn: Custom collate function (default handles variable channels with padding)
    
    Returns:
        DataLoader instance
    """
    dataset = HDF5WindowDataset(h5_dir_path, return_metadata=return_metadata)
    
    # Default collate function that handles variable channel counts
    if collate_fn is None:
        def default_collate(batch):
            """Collate function that pads to max channels in batch."""
            if dataset.return_metadata:
                signals, coords, regs, labels, metadata = zip(*batch)
            else:
                signals, coords, regs, labels = zip(*batch)
            
            # Find max channels in this batch
            max_ch = max(s.shape[0] for s in signals)
            n_timesteps = signals[0].shape[1]
            batch_size = len(signals)
            
            # Pad to max channels
            signals_padded = torch.zeros(batch_size, max_ch, n_timesteps)
            coords_padded = torch.zeros(batch_size, max_ch, 3)
            regs_padded = torch.zeros(batch_size, max_ch, dtype=torch.long)
            
            for i, (sig, coord, reg) in enumerate(zip(signals, coords, regs)):
                n_ch = sig.shape[0]
                signals_padded[i, :n_ch] = sig
                coords_padded[i, :n_ch] = coord
                regs_padded[i, :n_ch] = reg
            
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            if dataset.return_metadata:
                return signals_padded, coords_padded, regs_padded, labels_tensor, metadata
            else:
                return signals_padded, coords_padded, regs_padded, labels_tensor
        
        collate_fn = default_collate
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    return dataloader


# Example usage and testing
if __name__ == "__main__":
    import sys
    from config_benchmark import BenchmarkConfig
    
    print("="*60)
    print("Testing HDF5 Dataset (Per-Patient Files)")
    print("Data preprocessed following make_dataset.py pipeline")
    print("="*60)
    
    config = BenchmarkConfig()
    h5_dir = os.path.join(config.OUTPUT_DATA_DIR, "all_windows_per_patient")
    
    if not os.path.exists(h5_dir):
        print(f"\nError: HDF5 directory not found: {h5_dir}")
        print("Please run extract_all_windows.py first to generate the data.")
        sys.exit(1)
    
    # Test dataset
    print("\n1. Creating dataset...")
    dataset = HDF5WindowDataset(h5_dir, return_metadata=True)
    
    print(f"\n2. Dataset length: {len(dataset)}")
    
    print("\n3. Loading first sample...")
    signals, coords, regs, label, metadata = dataset[0]
    print(f"   Signals shape: {signals.shape}")
    print(f"   Signals dtype: {signals.dtype}")
    print(f"   Signals range: [{signals.min():.3f}, {signals.max():.3f}]")
    print(f"   Coords shape: {coords.shape}")
    print(f"   Regs shape: {regs.shape}")
    print(f"   Label: {label} ({'ictal' if label == 1 else 'interictal'})")
    print(f"   Patient ID: {metadata['patient_id']}")
    print(f"   Window info: {metadata['window_info']}")
    
    print("\n4. Creating DataLoader (batch_size=8, num_workers=2)...")
    dataloader = create_dataloader(
        h5_dir, 
        batch_size=8, 
        shuffle=True, 
        num_workers=2,
        return_metadata=False
    )
    
    print("\n5. Testing batch loading...")
    for i, batch in enumerate(dataloader):
        signals, coords, regs, labels = batch
        print(f"   Batch {i}:")
        print(f"     Signals: {signals.shape}, dtype: {signals.dtype}")
        print(f"     Coords: {coords.shape}")
        print(f"     Regs: {regs.shape}")
        print(f"     Labels: {labels.shape}, values: {labels.tolist()}")
        
        # Verify data properties
        if i == 0:
            print(f"     Signals range: [{signals.min():.3f}, {signals.max():.3f}]")
            print(f"     Signals std: {signals.std():.3f} (should be ~1 due to z-score norm)")
        
        if i >= 2:  # Only test 3 batches
            break
    
    print("\n✓ Dataset and DataLoader working correctly!")
    print("\nNote: Data is already normalized (per-channel z-score from make_dataset.py)")