"""
Extract embeddings from trained models for benchmark comparison.

This script loads a trained classifier and extracts CLS token embeddings
along with labels for comparison with other SOTA models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

from config_benchmark import BenchmarkConfig
from dataset_semiology import SemiologyDataset
from dataset_ictal_interictal import IctalInterictalDataset
from model_classifier import MultivarWav2Vec2Classifier

warnings.filterwarnings("ignore")


def extract_embeddings(model, dataloader, device='cuda'):
    """
    Extract embeddings from a model.
    
    Args:
        model: Trained classifier model
        dataloader: DataLoader for the dataset
        device: Device to run on
    
    Returns:
        embeddings: (N, embedding_dim) array of embeddings
        labels: (N,) array of class labels
        indices: (N,) array of sample indices
    """
    model.eval()
    model.to(device)
    
    all_embeddings = []
    all_labels = []
    all_indices = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            signals, regs, labels, indices = batch
            
            # Handle NaNs
            signals = torch.nan_to_num(signals, posinf=0.0, neginf=0.0)
            
            # Move to device
            signals = signals.to(device)
            regs = regs.to(device)
            
            # Extract embeddings
            embeddings = model.get_embeddings(signals, regs)
            
            # Store
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_indices.append(indices.cpu().numpy())
    
    # Concatenate
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    
    return embeddings, labels, indices


def load_model_from_checkpoint(checkpoint_path, num_classes):
    """
    Load a trained model from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of classes
    
    Returns:
        model: Loaded model
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model = MultivarWav2Vec2Classifier(num_classes=num_classes, freeze_encoder=False)
    
    # Load state dict (handle Lightning wrapper)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix from keys
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    print("✓ Model loaded successfully")
    return model


def extract_semiology_embeddings(config, checkpoint_path, output_file):
    """
    Extract embeddings for semiology classification.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to trained model checkpoint
        output_file: Output .npz file path
    """
    print("\n" + "=" * 60)
    print("Extracting Semiology Embeddings")
    print("=" * 60)
    
    # Create datasets
    train_dataset = SemiologyDataset(config=config, split='train')
    val_dataset = SemiologyDataset(config=config, split='val')
    test_dataset = SemiologyDataset(config=config, split='test')
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, train_dataset.num_classes)
    
    # Extract embeddings for each split
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = {}
    for split_name, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
        print(f"\n{split_name.upper()} set:")
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            persistent_workers=False
        )
        
        embeddings, labels, indices = extract_embeddings(model, dataloader, device)
        
        # Get patient IDs
        patient_ids = dataset.metadata.loc[indices, 'Patient'].values
        
        # Store
        results[f'{split_name}_embeddings'] = embeddings
        results[f'{split_name}_labels'] = labels
        results[f'{split_name}_indices'] = indices
        results[f'{split_name}_patient_ids'] = patient_ids
        
        print(f"  - Extracted {len(embeddings)} embeddings")
        print(f"  - Embedding dimension: {embeddings.shape[1]}")
    
    # Add label names
    results['label_names'] = dataset.get_label_names()
    results['num_classes'] = dataset.num_classes
    
    # Save
    np.savez(output_file, **results)
    print(f"\n✓ Saved embeddings to: {output_file}")


def extract_ictal_interictal_embeddings(config, checkpoint_path, output_file):
    """
    Extract embeddings for ictal/interictal classification.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to trained model checkpoint
        output_file: Output .npz file path
    """
    print("\n" + "=" * 60)
    print("Extracting Ictal/Interictal Embeddings")
    print("=" * 60)
    
    # Create datasets
    train_dataset = IctalInterictalDataset(config=config, split='train')
    val_dataset = IctalInterictalDataset(config=config, split='val')
    test_dataset = IctalInterictalDataset(config=config, split='test')
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, 2)
    
    # Extract embeddings for each split
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = {}
    for split_name, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
        print(f"\n{split_name.upper()} set:")
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            persistent_workers=False
        )
        
        embeddings, labels, indices = extract_embeddings(model, dataloader, device)
        
        # Get patient IDs
        patient_ids = dataset.all_patients[indices]
        
        # Store
        results[f'{split_name}_embeddings'] = embeddings
        results[f'{split_name}_labels'] = labels
        results[f'{split_name}_indices'] = indices
        results[f'{split_name}_patient_ids'] = patient_ids
        
        print(f"  - Extracted {len(embeddings)} embeddings")
        print(f"  - Embedding dimension: {embeddings.shape[1]}")
        print(f"  - Class distribution: Interictal={np.sum(labels==0)}, Ictal={np.sum(labels==1)}")
    
    # Add label names
    results['label_names'] = dataset.get_label_names()
    results['num_classes'] = 2
    
    # Save
    np.savez(output_file, **results)
    print(f"\n✓ Saved embeddings to: {output_file}")


def main(args):
    """Main function."""
    # Configuration
    config = BenchmarkConfig()
    config.create_directories()
    
    # Determine output file
    if args.output is None:
        if args.task == 'semiology':
            output_file = os.path.join(config.EMBEDDINGS_DIR, 'semiology_embeddings.npz')
        else:
            output_file = os.path.join(config.EMBEDDINGS_DIR, 'ictal_interictal_embeddings.npz')
    else:
        output_file = args.output
    
    # Extract embeddings
    if args.task == 'semiology':
        extract_semiology_embeddings(config, args.checkpoint, output_file)
    elif args.task == 'ictal_interictal':
        extract_ictal_interictal_embeddings(config, args.checkpoint, output_file)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    print("\n✓ Embedding extraction complete!")
    print(f"\nTo load embeddings in Python:")
    print(f"  import numpy as np")
    print(f"  data = np.load('{output_file}')")
    print(f"  train_embeddings = data['train_embeddings']")
    print(f"  train_labels = data['train_labels']")
    print(f"  label_names = data['label_names']")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings from trained models")
    
    parser.add_argument("task", type=str, choices=['semiology', 'ictal_interictal'],
                       help="Task type")
    parser.add_argument("checkpoint", type=str,
                       help="Path to trained model checkpoint (.ckpt file)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output .npz file path (default: auto-generated)")
    
    args = parser.parse_args()
    
    main(args)
