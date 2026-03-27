"""
PyTorch Dataset for lazy-loading windows from NPZ files.

This dataset loads windows on-the-fly using memory-mapped NPZ files, avoiding loading
the entire dataset into memory at once.

Data is already preprocessed and normalized during extraction (following make_dataset.py):
- Per-channel z-score normalization: (x - nanmean(x)) / nanstd(x)
- No additional normalization needed in dataset
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json


class NPZWindowDataset(Dataset):
    """
    PyTorch Dataset that lazy-loads windows from per-patient NPZ files using memory mapping.
    
    Each window contains:
    - signals: (n_channels, n_timesteps) EEG data
    - coords: (n_channels, 3) MNI coordinates
    - regs: (n_channels,) brain region labels
    - label: 0 (interictal) or 1 (ictal)
    - metadata: patient_id, channel_names, window_info
    """
    
    def __init__(self, npz_dir_path, transform=None, return_metadata=False, use_mmap=True):
        """
        Initialize the dataset from a directory of per-patient NPZ files.
        
        Note: Data is already normalized during extraction (per-channel z-score).
        Additional transforms should not include normalization.
        
        Args:
            npz_dir_path: Path to directory containing per-patient NPZ files
            transform: Optional transform to apply to signals (e.g., augmentation)
            return_metadata: If True, return metadata dict with each sample
            use_mmap: If True, use memory mapping (lazy loading). If False, load all data into RAM.
        """
        self.npz_dir_path = npz_dir_path
        self.transform = transform
        self.return_metadata = return_metadata
        self.use_mmap = use_mmap
        
        # Find all patient NPZ files
        from glob import glob
        self.patient_files = sorted(glob(os.path.join(npz_dir_path, "*.npz")))
        
        if len(self.patient_files) == 0:
            raise ValueError(f"No NPZ files found in {npz_dir_path}")
        
        print(f"Loading NPZ dataset from directory: {npz_dir_path}")
        print(f"  Found {len(self.patient_files)} patient files")
        
        # First pass: collect window counts and metadata
        window_counts = []
        patient_id_list = []
        total_interictal = 0
        total_ictal = 0
        
        for file_idx, patient_file in enumerate(self.patient_files):
            # Load with mmap_mode='r' for memory-mapped reading (no data loaded into RAM)
            data = np.load(patient_file, mmap_mode='r', allow_pickle=True)
            n_windows = data['signals'].shape[0]
            patient_id = str(data['patient_id'])
            
            window_counts.append(n_windows)
            patient_id_list.append(patient_id)
            
            # Count labels (only loads the labels array, not the full data)
            labels = data['labels']
            total_interictal += np.sum(labels == 0)
            total_ictal += np.sum(labels == 1)
        
        # Build index map efficiently (vectorized)
        # Pre-allocate arrays instead of repeated appends
        total_windows = sum(window_counts)
        
        # Use numpy for efficient index generation
        file_indices = np.repeat(np.arange(len(self.patient_files)), window_counts)
        window_indices = np.concatenate([np.arange(count) for count in window_counts])
        
        # Store as list of tuples (required for indexing)
        self.index_map = list(zip(file_indices.tolist(), window_indices.tolist()))
        
        # Replicate patient IDs efficiently using numpy repeat
        # Convert to array, repeat, then back to list (much faster than list comprehension)
        patient_id_array = np.array(patient_id_list, dtype=object)
        self.patient_ids = np.repeat(patient_id_array, window_counts).tolist()
        
        self.length = len(self.index_map)
        
        # Store label counts for class weight calculation
        self.label_counts = np.array([total_interictal, total_ictal])
        
        # Get dimensions from first file
        data = np.load(self.patient_files[0], mmap_mode='r')
        self.n_channels = data['signals'].shape[1]
        self.n_timesteps = data['signals'].shape[2]
        
        print(f"  Total windows: {self.length}")
        print(f"  Window shape: ({self.n_channels}, {self.n_timesteps})")
        print(f"  Interictal (0): {total_interictal} ({100*total_interictal/self.length:.1f}%)")
        print(f"  Ictal (1): {total_ictal} ({100*total_ictal/self.length:.1f}%)")
        
        # Load data based on mode
        if self.use_mmap:
            print(f"  Mode: Memory-mapped (lazy loading)")
            # Cache for memory-mapped file handles (one per worker)
            self._file_handles = {}
            self._data_cache = None
        else:
            print(f"  Mode: Loading all data into RAM...")
            self._file_handles = None
            self._data_cache = self._load_all_data()
            print(f"  ✓ All data loaded into RAM")
    
    def _load_all_data(self):
        """
        Load all data into RAM (non-memory-mapped mode).
        
        Returns:
            List of dicts, one per patient file, containing all data arrays
        """
        data_cache = []
        for file_idx, patient_file in enumerate(self.patient_files):
            # Load entire file into memory
            data = np.load(patient_file, allow_pickle=True)
            
            # Store as dict of arrays (convert to actual arrays, not memmap)
            patient_data = {
                'signals': np.array(data['signals']),
                'coords': np.array(data['coords']),
                'regs': np.array(data['regs']),
                'labels': np.array(data['labels']),
                'patient_id': data['patient_id'],
            }
            
            # Optional fields
            if 'n_channels' in data:
                patient_data['n_channels'] = np.array(data['n_channels'])
            if 'ch_names' in data:
                patient_data['ch_names'] = data['ch_names']
            if 'window_info' in data:
                patient_data['window_info'] = data['window_info']
            
            data_cache.append(patient_data)
            
        return data_cache
    
    def _get_npz_file(self, file_idx):
        """
        Get or create memory-mapped NPZ file handle for a specific patient file.
        Each worker needs its own file handles for thread safety.
        
        Args:
            file_idx: Index of patient file in self.patient_files
        
        Returns:
            numpy NpzFile handle with memory mapping
        """
        if not self.use_mmap:
            # Return the in-memory cache
            return self._data_cache[file_idx]
        
        if file_idx not in self._file_handles:
            self._file_handles[file_idx] = np.load(
                self.patient_files[file_idx], 
                mmap_mode='r',
                allow_pickle=True
            )
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
        
        # Get memory-mapped file handle for this patient
        f = self._get_npz_file(file_idx)
        
        # Load data for this index (memory-mapped, only reads this window from disk)
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
            ch_names = json.loads(str(f['ch_names'][window_idx]))
            window_info = json.loads(str(f['window_info'][window_idx]))
            
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
        f = self._get_npz_file(file_idx)
        return int(f['labels'][window_idx])
    
    def get_patient_id(self, idx):
        """Get just the patient ID for a given index."""
        return self.patient_ids[idx]
    
    def get_labels(self):
        """Get all labels (loads from all patient files)."""
        labels = []
        for file_idx, patient_file in enumerate(self.patient_files):
            data = np.load(patient_file, mmap_mode='r')
            labels.extend(data['labels'][:].tolist())
        return np.array(labels)
    
    def get_patient_ids(self):
        """Get all patient IDs (already loaded in __init__)."""
        return self.patient_ids
    
    def get_class_weights(self):
        """
        Calculate class weights for handling imbalanced datasets.
        Uses pre-computed label counts from initialization.
        
        Returns:
            torch.FloatTensor: Class weights for each class
        """
        total_samples = self.length
        num_classes = len(self.label_counts)
        class_weights = torch.FloatTensor([
            total_samples / (num_classes * count) 
            for count in self.label_counts
        ])
        return class_weights
    
    def __del__(self):
        """Close all NPZ file handles when dataset is destroyed."""
        if self.use_mmap and self._file_handles is not None:
            for f in self._file_handles.values():
                if f is not None:
                    f.close()


def create_dataloader(npz_dir_path, batch_size=32, shuffle=True, num_workers=4, 
                      return_metadata=False, pin_memory=True, collate_fn=None, use_mmap=True):
    """
    Create a PyTorch DataLoader for the NPZ dataset.
    
    Args:
        npz_dir_path: Path to directory containing per-patient NPZ files
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        return_metadata: Whether to return metadata with each sample
        pin_memory: Pin memory for faster GPU transfer
        collate_fn: Custom collate function (default handles variable channels with padding)
        use_mmap: If True, use memory mapping. If False, load all data into RAM.
    
    Returns:
        DataLoader instance
    """
    dataset = NPZWindowDataset(npz_dir_path, return_metadata=return_metadata, use_mmap=use_mmap)
    
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
    print("Testing NPZ Dataset (Per-Patient Files)")
    print("Data preprocessed following make_dataset.py pipeline")
    print("Using memory-mapped loading for efficiency")
    print("="*60)
    
    config = BenchmarkConfig()
    npz_dir = os.path.join(config.OUTPUT_DATA_DIR, "all_windows_per_patient")
    
    if not os.path.exists(npz_dir):
        print(f"\nError: NPZ directory not found: {npz_dir}")
        print("Please run extract_all_windows.py first to generate the data.")
        sys.exit(1)
    
    # Test dataset
    print("\n1. Creating dataset...")
    dataset = NPZWindowDataset(npz_dir, return_metadata=True)
    
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
    
    print("\n4. Creating DataLoader (batch_size=8, num_workers=0)...")
    dataloader = create_dataloader(
        npz_dir, 
        batch_size=8, 
        shuffle=True, 
        num_workers=0,  # Use 0 for testing to avoid Windows multiprocessing issues
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