"""
Extract embeddings from trained WVNT and Conformer models on seizure data.

This script replicates the embedding extraction from the notebook experiment,
using trained benchmark models instead of the SWaV model.
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

from models.wrappers import WVNTChannelAggregator, ConformerChannelAggregator
from lightning_modules.ictal_classifier import IctalClassifier


def load_seizure_data(pickle_path, metadata_path):
    """
    Load seizure data from pickle file.
    
    Returns:
        signals: List of [n_channels, timesteps] arrays
        coords: List of [n_channels, 3] arrays
        regs: List of [n_channels] arrays
        ch_names: List of channel name lists
        metadata: DataFrame with seizure metadata
    """
    print(f"Loading seizure data from {pickle_path}...")
    
    with open(pickle_path, 'rb') as f:
        # Pickle file contains: signals, coords, regs, ch_names, indices
        signals, coords, regs, ch_names, indices = pickle.load(f)
    
    print(f"  ✓ Loaded {len(signals)} seizures (as lists of numpy arrays)")
    
    # Load metadata (pre-constructed for these seizures)
    print(f"Loading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path)
    
    # Verify alignment - metadata should have same or close to same number of rows
    if len(metadata) != len(signals):
        print(f"  ⚠ Warning: Metadata has {len(metadata)} rows but we have {len(signals)} seizures")
        print(f"    Using first {min(len(metadata), len(signals))} seizures")
        # Trim to matching length
        n = min(len(metadata), len(signals))
        signals = signals[:n]
        coords = coords[:n]
        regs = regs[:n]
        ch_names = ch_names[:n]
        metadata = metadata.iloc[:n].reset_index(drop=True)
    
    print(f"  ✓ Final dataset size: {len(signals)} seizures")
    print(f"  ✓ Metadata shape: {metadata.shape}")
    print(f"  ✓ Metadata columns: {list(metadata.columns)[:10]}...")  # Show first 10
    
    return signals, coords, regs, ch_names, metadata


def load_model_checkpoint(model_name, aggregation, checkpoint_dir):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_name: 'wvnt' or 'conformer'
        aggregation: 'mean' or 'max'
        checkpoint_dir: Base directory containing checkpoints
    
    Returns:
        model: Loaded Lightning model in eval mode
    """
    checkpoint_base = Path(checkpoint_dir) / f'{model_name}_{aggregation}' / 'best'
    
    print(f"\nLoading {model_name} ({aggregation}) checkpoint...")
    print(f"  Searching in: {checkpoint_base}")
    
    if not checkpoint_base.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_base}")
    
    # Find checkpoint files
    ckpt_files = list(checkpoint_base.glob('*.ckpt'))
    
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_base}")
    
    # Prioritize versioned checkpoints (last-v2.ckpt > last-v1.ckpt > last.ckpt)
    versioned_ckpts = []
    for ckpt_file in ckpt_files:
        filename = ckpt_file.name
        import re
        match = re.search(r'last-v(\d+)\.ckpt', filename)
        if match:
            version = int(match.group(1))
            versioned_ckpts.append((version, str(ckpt_file)))
    
    if versioned_ckpts:
        versioned_ckpts.sort(reverse=True)
        ckpt_path = versioned_ckpts[0][1]
        print(f"  Using versioned checkpoint: {Path(ckpt_path).name}")
    else:
        # Fall back to last.ckpt
        last_ckpt = checkpoint_base / 'last.ckpt'
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)
            print(f"  Using: last.ckpt")
        else:
            ckpt_path = str(ckpt_files[0])
            print(f"  Using first available: {Path(ckpt_path).name}")
    
    # Create base model architecture
    if model_name == 'wvnt':
        base_model = WVNTChannelAggregator(
            checkpoint_path=None,
            aggregation=aggregation,
            freeze_backbone=False
        )
    elif model_name == 'conformer':
        base_model = ConformerChannelAggregator(
            emb_size=32,
            depth=3,
            aggregation=aggregation,
            freeze_backbone=False
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load from checkpoint
    model = IctalClassifier.load_from_checkpoint(
        ckpt_path,
        model=base_model,
        strict=False
    )
    
    model.eval()
    print(f"  ✓ Model loaded successfully")
    
    return model


def extract_embeddings_from_model(model, signals, regs, device, batch_size=32):
    """
    Extract embeddings from a model for all seizures.
    
    Args:
        model: Loaded Lightning model
        signals: List of [n_channels, timesteps] numpy arrays
        regs: List of [n_channels] numpy arrays
        device: torch device
        batch_size: Batch size for inference
    
    Returns:
        embeddings: [n_seizures, embed_dim] numpy array
    """
    model = model.to(device)
    model.eval()
    
    all_embeddings = []
    
    print(f"  Extracting embeddings...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(signals), batch_size)):
            batch_signals = signals[i:i+batch_size]
            batch_regs = regs[i:i+batch_size]
            
            # Find max channels in this batch
            max_ch = max(s.shape[0] for s in batch_signals)
            timesteps = batch_signals[0].shape[1]
            
            # Pad to max channels
            signals_padded = torch.zeros(len(batch_signals), max_ch, timesteps)
            regs_padded = torch.zeros(len(batch_signals), max_ch, dtype=torch.long)
            
            for j, (sig, reg) in enumerate(zip(batch_signals, batch_regs)):
                n_ch = sig.shape[0]
                signals_padded[j, :n_ch] = torch.from_numpy(sig).float()
                regs_padded[j, :n_ch] = torch.from_numpy(reg).long()
            
            # Move to device
            signals_padded = signals_padded.to(device)
            regs_padded = regs_padded.to(device)
            
            # Extract embeddings
            # Check if model needs regs (WVNT needs it, Conformer doesn't)
            if hasattr(model.model, 'get_embeddings'):
                # Call get_embeddings on the wrapper
                try:
                    # Try with regs first (for WVNT)
                    embeddings = model.model.get_embeddings(signals_padded, regs_padded)
                except TypeError:
                    # Fall back to no regs (for Conformer)
                    embeddings = model.model.get_embeddings(signals_padded)
            else:
                raise AttributeError("Model does not have get_embeddings method")
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  ✓ Extracted embeddings: {all_embeddings.shape}")
    
    return all_embeddings


def compute_soz_labels(signals, regs, ch_names, metadata, soz_csv_path, unique_regs_path):
    """
    Compute onset location labels from SOZ metadata.
    
    Replicates the label derivation from notebook cell 99.
    
    Args:
        signals: List of signal arrays
        regs: List of region arrays
        ch_names: List of channel name lists
        metadata: Seizure metadata DataFrame
        soz_csv_path: Path to soz_electrodes.csv
        unique_regs_path: Path to unique_regs.csv
    
    Returns:
        labels_dict: Dictionary mapping label names to list of seizure indices
        valid_indices: List of seizure indices that have SOZ labels
    """
    print("\nComputing SOZ-based labels...")
    
    # Load SOZ electrodes
    soz_elecs = pd.read_csv(soz_csv_path)
    print(f"  ✓ Loaded SOZ electrodes: {len(soz_elecs)} entries")
    
    # Load unique regions
    unique_regs = pd.read_csv(unique_regs_path)
    print(f"  ✓ Loaded unique regions: {len(unique_regs)} regions")
    
    # Create is_soz array [n_seizures, max_channels]
    max_channels = max(s.shape[0] for s in signals)
    is_soz = np.zeros((len(signals), max_channels), dtype=bool)
    
    # Mark SOZ channels for each seizure
    for pt, group in soz_elecs.groupby('rid'):
        soz_ch_names = group['soz_electrode'].values
        
        # Find seizures for this patient
        pt_idx = metadata.reset_index(drop=True).query('Patient == @pt').index
        if len(pt_idx) == 0:
            continue
        
        # Get channel names for these seizures
        pt_ch_names = [ch_names[i] for i in pt_idx]
        
        # Mark SOZ channels
        for idx, names in zip(pt_idx, pt_ch_names):
            for ch_idx, ch_name in enumerate(names):
                if '-' in ch_name:
                    ch_name1 = ch_name.split('-')[0]
                    ch_name2 = ch_name.split('-')[1]
                    
                    if (ch_name1 in soz_ch_names) or (ch_name2 in soz_ch_names):
                        is_soz[idx, ch_idx] = True
    
    # Compute labels based on SOZ regions
    labels_dict = {
        "Left Frontal": [],
        "Right Frontal": [],
        "Left Temporal": [],
        "Right Temporal": [],
    }
    
    valid_indices = []
    
    for i in range(len(signals)):
        # Get actual number of channels for this seizure
        n_ch = regs[i].shape[0]
        
        # Get SOZ channels (only for actual channels, not padded ones)
        soz_mask = is_soz[i, :n_ch]
        soz_regs_sample = regs[i][soz_mask]
        
        if len(soz_regs_sample) == 0:
            continue
        
        # Map region indices to region names
        soz_regs_sample = unique_regs.iloc[soz_regs_sample - 1].reg.values
        
        # Count by category
        n_lf = sum(["Frontal" in reg and "L" in reg for reg in soz_regs_sample])
        n_rf = sum(["Frontal" in reg and "R" in reg for reg in soz_regs_sample])
        n_lt = sum(["Temporal" in reg and "L" in reg for reg in soz_regs_sample])
        n_rt = sum(["Temporal" in reg and "R" in reg for reg in soz_regs_sample])
        
        # Assign label based on max count
        max_idx = np.argmax([n_lf, n_rf, n_lt, n_rt])
        if max_idx == 0:
            labels_dict["Left Frontal"].append(i)
            valid_indices.append(i)
        elif max_idx == 1:
            labels_dict["Right Frontal"].append(i)
            valid_indices.append(i)
        elif max_idx == 2:
            labels_dict["Left Temporal"].append(i)
            valid_indices.append(i)
        elif max_idx == 3:
            labels_dict["Right Temporal"].append(i)
            valid_indices.append(i)
    
    print(f"  ✓ Label counts:")
    for label, indices in labels_dict.items():
        print(f"    {label}: {len(indices)}")
    print(f"  ✓ Total labeled seizures: {len(valid_indices)}")
    
    return labels_dict, valid_indices


def save_embeddings(embeddings, labels_dict, valid_indices, metadata, output_path):
    """
    Save embeddings and labels to npz file.
    
    Args:
        embeddings: [n_seizures, embed_dim] array
        labels_dict: Dictionary mapping label names to indices
        valid_indices: List of valid seizure indices
        metadata: Seizure metadata DataFrame
        output_path: Path to save .npz file
    """
    # Filter to valid indices
    embeddings_filtered = embeddings[valid_indices]
    
    # Create labels array
    labels = np.empty(len(valid_indices), dtype='<U20')  # Unicode strings
    for label_name, indices in labels_dict.items():
        for idx in indices:
            # Find position in valid_indices
            pos = valid_indices.index(idx)
            labels[pos] = label_name
    
    # Filter metadata
    metadata_filtered = metadata.iloc[valid_indices].reset_index(drop=True)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        embeddings=embeddings_filtered,
        labels=labels,
        patient_ids=metadata_filtered['Patient'].values,
        seizure_ids=metadata_filtered.index.values
    )
    
    print(f"  ✓ Saved to {output_path}")
    print(f"    Embeddings shape: {embeddings_filtered.shape}")
    print(f"    Labels shape: {labels.shape}")


def main():
    """Main extraction pipeline."""
    
    # Paths
    base_dir = Path(__file__).parent.parent
    pickle_path = base_dir / 'data' / 'all_sz_coords_data_spatiotemporal_regs_10s.pkl'
    metadata_path = base_dir / 'data' / 'gui_data' / 'master_metadata.csv'  # Pre-constructed metadata for the seizures
    soz_csv_path = base_dir / 'data' / 'metadata' / 'soz_electrodes.csv'
    unique_regs_path = base_dir / 'data' / 'unique_regs.csv'
    checkpoint_dir = base_dir / 'checkpoints'
    output_dir = base_dir / 'benchmark' / 'embeddings'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load seizure data
    signals, coords, regs, ch_names, metadata = load_seizure_data(pickle_path, metadata_path)
    
    # Compute SOZ labels (same for all models)
    labels_dict, valid_indices = compute_soz_labels(
        signals, regs, ch_names, metadata, soz_csv_path, unique_regs_path
    )
    
    # Extract embeddings for each model
    models_to_extract = [
        ('wvnt', 'mean'),
        ('conformer', 'mean')
    ]
    
    for model_name, aggregation in models_to_extract:
        print(f"\n{'='*80}")
        print(f"Processing {model_name.upper()} ({aggregation} aggregation)")
        print('='*80)
        
        try:
            # Load model
            model = load_model_checkpoint(model_name, aggregation, checkpoint_dir)
            
            # Extract embeddings
            embeddings = extract_embeddings_from_model(
                model, signals, regs, device, batch_size=32
            )
            
            # Save
            output_path = output_dir / f'{model_name}_{aggregation}_embeddings.npz'
            save_embeddings(embeddings, labels_dict, valid_indices, metadata, output_path)
            
        except Exception as e:
            print(f"  ✗ Failed to process {model_name}_{aggregation}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("Embedding extraction complete!")
    print('='*80)


if __name__ == '__main__':
    main()
