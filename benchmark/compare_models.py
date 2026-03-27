"""
Model Comparison Script for Benchmark Models.

Evaluates and compares all trained models (ST, ConformerRegions, Conformer, WVNT)
on validation and test sets with bootstrap confidence intervals.

Usage:
    python compare_models.py
    python compare_models.py --data-dir data/all_windows_per_patient_global_norm
    python compare_models.py --n-bootstrap 2000 --output results/comparison.csv
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

from dataset_all_windows import NPZWindowDataset
from lightning_modules.ictal_classifier import IctalClassifier
from config_benchmark import BenchmarkConfig
from models.wrappers import (
    ONSETChannelAggregator,
    WVNTChannelAggregator,
    ConformerChannelAggregator,
    ConformerRegionsChannelAggregator,
    STChannelAggregator
)


def find_best_checkpoint(model_name: str, aggregation: str, checkpoint_base_dir: str = 'checkpoints') -> Optional[str]:
    """
    Find the best checkpoint for a model.
    
    Args:
        model_name: Model name (e.g., 'st', 'conformer_regions', 'wvnt')
        aggregation: Aggregation method ('mean' or 'max')
        checkpoint_base_dir: Base directory for checkpoints
    
    Returns:
        Path to best checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_base_dir) / f'{model_name}_{aggregation}' / 'best'
    
    print(f"  Searching in: {checkpoint_dir}")
    
    if not checkpoint_dir.exists():
        print(f"  ✗ Directory does not exist")
        return None
    
    # Find all checkpoint files
    ckpt_files = list(checkpoint_dir.glob('*.ckpt'))
    
    print(f"  Found {len(ckpt_files)} checkpoint files")
    if ckpt_files:
        for ckpt in ckpt_files:
            print(f"    - {ckpt.name}")
    
    if not ckpt_files:
        print(f"  ✗ No .ckpt files found")
        return None
    
    # Try to find checkpoint with highest AUROC in filename
    # Pattern: epoch=XX-val/auroc=0.XXX.ckpt or epoch=XX-val_auroc=0.XXX.ckpt
    best_ckpt = None
    best_auroc = -1
    
    for ckpt_file in ckpt_files:
        filename = ckpt_file.name
        
        # Extract AUROC from filename
        # Try both formats: val/auroc and val_auroc
        match = re.search(r'val[/_]auroc[=_]([\d.]+)', filename)
        if match:
            auroc = float(match.group(1))
            if auroc > best_auroc:
                best_auroc = auroc
                best_ckpt = str(ckpt_file)
    
    # If no AUROC in filename, look for versioned checkpoints
    if best_ckpt is None:
        # Find all last-vN.ckpt files
        versioned_ckpts = []
        for ckpt_file in ckpt_files:
            filename = ckpt_file.name
            # Pattern: last-v1.ckpt, last-v2.ckpt, etc.
            match = re.search(r'last-v(\d+)\.ckpt', filename)
            if match:
                version = int(match.group(1))
                versioned_ckpts.append((version, str(ckpt_file)))
        
        # Use highest version if any versioned checkpoints exist
        if versioned_ckpts:
            versioned_ckpts.sort(reverse=True)  # Sort by version descending
            best_version, best_ckpt = versioned_ckpts[0]
            print(f"  Using versioned checkpoint: last-v{best_version}.ckpt")
        else:
            # Fall back to last.ckpt if no versioned files
            last_ckpt = checkpoint_dir / 'last.ckpt'
            if last_ckpt.exists():
                best_ckpt = str(last_ckpt)
                print(f"  Using: last.ckpt")
    
    # If still nothing, just use the first checkpoint
    if best_ckpt is None and ckpt_files:
        best_ckpt = str(ckpt_files[0])
        print(f"  Using first available: {Path(best_ckpt).name}")
    
    return best_ckpt


def evaluate_model(
    model: IctalClassifier,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataloader.
    
    Args:
        model: Lightning model
        dataloader: DataLoader for evaluation
        device: Device to run on
    
    Returns:
        (probs, labels): Predicted probabilities and true labels
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    # Check if model needs regions
    needs_regions = (
        hasattr(model.model, 'st_model') or 
        hasattr(model.model, 'conformer_regions') or
        'Regions' in model.model.__class__.__name__ or
        'ST' in model.model.__class__.__name__ or
        'WVNT' in model.model.__class__.__name__
    )
    
    with torch.no_grad():
        for batch in dataloader:
            signals, coords, regs, labels = batch
            signals = signals.to(device)
            regs = regs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            if needs_regions:
                logits = model(signals, regs)
            else:
                logits = model(signals)
            
            probs = F.softmax(logits, dim=1)[:, 1]  # P(ictal)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    return all_probs, all_labels


def bootstrap_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    ci: int = 95,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute metrics with bootstrap confidence intervals.
    
    Args:
        probs: Predicted probabilities [n_samples]
        labels: True labels [n_samples]
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval percentage
        random_seed: Random seed for reproducibility
    
    Returns:
        (mean_metrics, ci_lower, ci_upper): Each is [f1, auroc, auprc]
    """
    np.random.seed(random_seed)
    
    n_samples = len(labels)
    bootstrap_results = []
    
    for i in range(n_bootstrap):
        # Bootstrap sampling
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_probs = probs[indices]
        boot_labels = labels[indices]
        
        # Skip if bootstrap sample has only one class
        if len(np.unique(boot_labels)) < 2:
            continue
        
        # Compute metrics
        preds = (boot_probs > 0.5).astype(int)
        
        try:
            f1 = f1_score(boot_labels, preds)
            auroc = roc_auc_score(boot_labels, boot_probs)
            auprc = average_precision_score(boot_labels, boot_probs)
            
            bootstrap_results.append([f1, auroc, auprc])
        except:
            # Skip this bootstrap sample if metrics can't be computed
            continue
    
    bootstrap_results = np.array(bootstrap_results)
    
    # Compute confidence intervals
    mean_metrics = np.mean(bootstrap_results, axis=0)
    ci_lower = np.percentile(bootstrap_results, (100 - ci) / 2, axis=0)
    ci_upper = np.percentile(bootstrap_results, 50 + ci / 2, axis=0)
    
    return mean_metrics, ci_lower, ci_upper


def print_comparison_table(df: pd.DataFrame):
    """
    Print formatted comparison table.
    
    Args:
        df: DataFrame with comparison results
    """
    print("\n" + "="*100)
    print(" " * 35 + "Model Comparison Results")
    print("="*100)
    
    # Check if DataFrame is empty
    if len(df) == 0:
        print("\n⚠ No models were evaluated successfully. Check that:")
        print("  1. Model checkpoints exist in the checkpoint directory")
        print("  2. Checkpoint directory path is correct")
        print("  3. Models were trained with the specified aggregation method")
        return
    
    for split in ['validation', 'test']:
        split_df = df[df['split'] == split]
        
        if len(split_df) == 0:
            continue
        
        print(f"\n{split.capitalize()} Set:")
        print("-" * 100)
        print(f"{'Model':<25} {'F1 Score':<30} {'AUROC':<30} {'AUPRC':<30}")
        print("-" * 100)
        
        for _, row in split_df.iterrows():
            model_name = f"{row['model']} ({row['aggregation']})"
            
            # Calculate ± error from confidence intervals
            f1_err = (row['f1_ci_high'] - row['f1_ci_low']) / 2
            auroc_err = (row['auroc_ci_high'] - row['auroc_ci_low']) / 2
            auprc_err = (row['auprc_ci_high'] - row['auprc_ci_low']) / 2
            
            f1_str = f"{row['f1_mean']:.4f} ± {f1_err:.4f}"
            auroc_str = f"{row['auroc_mean']:.4f} ± {auroc_err:.4f}"
            auprc_str = f"{row['auprc_mean']:.4f} ± {auprc_err:.4f}"
            
            print(f"{model_name:<25} {f1_str:<30} {auroc_str:<30} {auprc_str:<30}")
        
        print("-" * 100)


def compare_all_models(
    models_config: List[Tuple[str, str]],
    data_dir: str,
    output_file: str,
    n_bootstrap: int = 1000,
    ci: int = 95,
    batch_size: int = 32,
    checkpoint_base_dir: str = 'checkpoints'
):
    """
    Compare all models on validation and test sets.
    
    Args:
        models_config: List of (model_name, aggregation) tuples
        data_dir: Path to NPZ data directory
        output_file: Path to output CSV file
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval percentage
        batch_size: Batch size for evaluation
        checkpoint_base_dir: Base directory for checkpoints
    """
    print("="*100)
    print(" " * 35 + "Model Comparison Script")
    print("="*100)
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint directory: {checkpoint_base_dir}")
    print(f"Output file: {output_file}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print(f"Confidence interval: {ci}%")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Show available checkpoint directories
    checkpoint_base = Path(checkpoint_base_dir)
    if checkpoint_base.exists():
        available_dirs = [d.name for d in checkpoint_base.iterdir() if d.is_dir()]
        if available_dirs:
            print(f"Available checkpoint directories:")
            for dirname in sorted(available_dirs):
                print(f"  - {dirname}")
            print()
        else:
            print(f"⚠ Warning: Checkpoint directory exists but is empty: {checkpoint_base}")
            print()
    else:
        print(f"⚠ Warning: Checkpoint directory does not exist: {checkpoint_base}")
        print()
    
    # ⚠️ IMPORTANT: Check that at least one checkpoint exists BEFORE loading data
    print("="*100)
    print("Checking for model checkpoints...")
    print("="*100)
    
    found_checkpoints = {}
    for model_name, aggregation in models_config:
        print(f"\nChecking {model_name.upper()} ({aggregation}):")
        ckpt_path = find_best_checkpoint(model_name, aggregation, checkpoint_base_dir)
        if ckpt_path:
            found_checkpoints[(model_name, aggregation)] = ckpt_path
            print(f"  ✓ Checkpoint found")
        else:
            print(f"  ✗ No checkpoint found")
    
    if not found_checkpoints:
        print("\n" + "="*100)
        print("⚠ ERROR: No checkpoints found for any requested models!")
        print("="*100)
        print("\nPlease check:")
        print(f"  1. Checkpoint directory path: {checkpoint_base_dir}")
        print(f"  2. Model names: {[m[0] for m in models_config]}")
        print(f"  3. Aggregation method: {aggregation}")
        print(f"\nExpected checkpoint structure:")
        print(f"  {checkpoint_base_dir}/{{model}}_{{aggregation}}/best/*.ckpt")
        print(f"\nExample:")
        print(f"  {checkpoint_base_dir}/st_mean/best/last.ckpt")
        print("\nExiting without loading dataset (no models to evaluate).\n")
        return
    
    print("\n" + "="*100)
    print(f"✓ Found {len(found_checkpoints)} checkpoint(s) - proceeding with evaluation")
    print("="*100)
    print()
    
    # Now load dataset ONCE (this is slow, so we only do it if checkpoints exist!)
    print("="*100)
    print("Loading dataset (this may take a while)...")
    print("="*100)
    
    dataset = NPZWindowDataset(str(data_dir), return_metadata=False, use_mmap=False)
    print(f"✓ Loaded {len(dataset)} windows\n")
    
    # Create train/val/test splits (same as train_benchmarks.py)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✓ Split dataset:")
    print(f"  Train: {len(train_dataset)} windows")
    print(f"  Validation: {len(val_dataset)} windows")
    print(f"  Test: {len(test_dataset)} windows\n")
    
    # Create dataloaders ONCE (reused for all models)
    def collate_fn(batch):
        """Collate function that handles variable channels with padding."""
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
        
        return signals_padded, coords_padded, regs_padded, labels_tensor
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print("✓ Created dataloaders\n")
    
    # Now iterate through models (data already loaded!)
    results = []
    
    for model_name, aggregation in models_config:
        # Skip if no checkpoint was found in the pre-check
        if (model_name, aggregation) not in found_checkpoints:
            continue
        
        print("="*100)
        print(f"Evaluating {model_name.upper()} ({aggregation} aggregation)")
        print("="*100)
        
        # Use the checkpoint path we already found
        ckpt_path = found_checkpoints[(model_name, aggregation)]
        print(f"✓ Using checkpoint: {ckpt_path}")
        
        # Load model - need to recreate architecture first
        try:
            # Create model architecture based on model name
            if model_name == 'onset':
                base_model = ONSETChannelAggregator(
                    checkpoint_path=None,
                    aggregation=aggregation,
                    freeze_backbone=False
                )
            elif model_name == 'wvnt':
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
            elif model_name == 'conformer_regions':
                base_model = ConformerRegionsChannelAggregator(
                    emb_size=32,
                    depth=3,
                    num_heads=8,
                    aggregation=aggregation,
                    freeze_backbone=False,
                    max_channels=150,
                    max_regions=42
                )
            elif model_name == 'st':
                base_model = STChannelAggregator(
                    spatial_embed_dim=32,
                    temporal_embed_dim=32,
                    depth_spatial=3,
                    depth_temporal=3,
                    n_spatial_tokens=20,
                    aggregation=aggregation,
                    freeze_backbone=False,
                    max_channels=150,
                    n_frames=20
                )
            else:
                print(f"✗ Unknown model: {model_name}\n")
                continue
            
            # Load checkpoint with model architecture
            # Use strict=False to allow loading even if checkpoint has extra keys (e.g., criterion.weight)
            model = IctalClassifier.load_from_checkpoint(
                ckpt_path,
                model=base_model,
                strict=False
            )
            model.eval()
            model = model.to(device)
            print(f"✓ Model loaded successfully\n")
        except Exception as e:
            print(f"✗ Failed to load model: {e}\n")
            import traceback
            traceback.print_exc()
            continue
        
        # Evaluate on validation set
        print("Evaluating on validation set...")
        try:
            val_probs, val_labels = evaluate_model(model, val_loader, device)
            print(f"  ✓ Inference complete: {len(val_labels)} samples")
            
            print(f"  Computing bootstrap confidence intervals (n={n_bootstrap})...")
            val_mean, val_ci_low, val_ci_high = bootstrap_metrics(
                val_probs, val_labels, n_bootstrap=n_bootstrap, ci=ci
            )
            print(f"  ✓ Validation F1: {val_mean[0]:.4f} ({val_ci_low[0]:.4f}-{val_ci_high[0]:.4f})")
            print(f"  ✓ Validation AUROC: {val_mean[1]:.4f} ({val_ci_low[1]:.4f}-{val_ci_high[1]:.4f})")
            print(f"  ✓ Validation AUPRC: {val_mean[2]:.4f} ({val_ci_low[2]:.4f}-{val_ci_high[2]:.4f})")
        except Exception as e:
            print(f"  ✗ Validation failed: {e}")
            val_mean = val_ci_low = val_ci_high = [np.nan, np.nan, np.nan]
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        try:
            test_probs, test_labels = evaluate_model(model, test_loader, device)
            print(f"  ✓ Inference complete: {len(test_labels)} samples")
            
            print(f"  Computing bootstrap confidence intervals (n={n_bootstrap})...")
            test_mean, test_ci_low, test_ci_high = bootstrap_metrics(
                test_probs, test_labels, n_bootstrap=n_bootstrap, ci=ci
            )
            print(f"  ✓ Test F1: {test_mean[0]:.4f} ({test_ci_low[0]:.4f}-{test_ci_high[0]:.4f})")
            print(f"  ✓ Test AUROC: {test_mean[1]:.4f} ({test_ci_low[1]:.4f}-{test_ci_high[1]:.4f})")
            print(f"  ✓ Test AUPRC: {test_mean[2]:.4f} ({test_ci_low[2]:.4f}-{test_ci_high[2]:.4f})")
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            test_mean = test_ci_low = test_ci_high = [np.nan, np.nan, np.nan]
        
        # Store results
        results.append({
            'model': model_name,
            'aggregation': aggregation,
            'split': 'validation',
            'f1_mean': val_mean[0],
            'f1_ci_low': val_ci_low[0],
            'f1_ci_high': val_ci_high[0],
            'auroc_mean': val_mean[1],
            'auroc_ci_low': val_ci_low[1],
            'auroc_ci_high': val_ci_high[1],
            'auprc_mean': val_mean[2],
            'auprc_ci_low': val_ci_low[2],
            'auprc_ci_high': val_ci_high[2],
        })
        
        results.append({
            'model': model_name,
            'aggregation': aggregation,
            'split': 'test',
            'f1_mean': test_mean[0],
            'f1_ci_low': test_ci_low[0],
            'f1_ci_high': test_ci_high[0],
            'auroc_mean': test_mean[1],
            'auroc_ci_low': test_ci_low[1],
            'auroc_ci_high': test_ci_high[1],
            'auprc_mean': test_mean[2],
            'auprc_ci_low': test_ci_low[2],
            'auprc_ci_high': test_ci_high[2],
        })
        
        print()
    
    # Save to CSV
    df = pd.DataFrame(results)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("="*100)
    print(f"✓ Results saved to {output_file}")
    print("="*100)
    
    # Print formatted table
    print_comparison_table(df)
    
    print("\n" + "="*100)
    print(" " * 40 + "Evaluation Complete!")
    print("="*100)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare benchmark models with bootstrap confidence intervals',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to NPZ data directory (default: from config_benchmark.py)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model_comparison_results.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['st', 'conformer_regions', 'conformer', 'wvnt'],
        help='Models to compare'
    )
    parser.add_argument(
        '--aggregation',
        type=str,
        default='mean',
        choices=['mean', 'max'],
        help='Aggregation method'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=1000,
        help='Number of bootstrap iterations'
    )
    parser.add_argument(
        '--confidence-interval',
        type=int,
        default=95,
        help='Confidence interval percentage'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='../checkpoints',
        help='Base directory for model checkpoints (default: ../checkpoints relative to benchmark/)'
    )
    
    args = parser.parse_args()
    
    # Get data directory from config if not specified
    if args.data_dir is None:
        config = BenchmarkConfig()
        args.data_dir = Path(config.OUTPUT_DATA_DIR) / 'all_windows_per_patient_perpatient'
    
    # Create models config
    models_config = [(model, args.aggregation) for model in args.models]
    
    # Run comparison
    compare_all_models(
        models_config=models_config,
        data_dir=args.data_dir,
        output_file=args.output,
        n_bootstrap=args.n_bootstrap,
        ci=args.confidence_interval,
        batch_size=args.batch_size,
        checkpoint_base_dir=args.checkpoint_dir
    )


if __name__ == '__main__':
    main()
