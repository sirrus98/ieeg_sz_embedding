"""
Evaluation script for benchmark models.

Loads trained model checkpoints and evaluates them on the test set.
Generates comprehensive metrics, confusion matrices, and ROC curves.

Usage:
    python evaluate_benchmarks.py --checkpoint checkpoints/onset_mean/best/epoch=28-val_auroc=0.891.ckpt
    python evaluate_benchmarks.py --checkpoint checkpoints/wvnt_max/best/last.ckpt --output-dir results/wvnt_max
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

from dataset_all_windows import NPZWindowDataset
from lightning_modules.ictal_classifier import IctalClassifier


def evaluate_model(
    checkpoint_path: str,
    npz_dir: str,
    output_dir: str = 'evaluation_results',
    batch_size: int = 32
):
    """
    Evaluate a trained model on test data.
    
    Args:
        checkpoint_path: Path to model checkpoint
        npz_dir: Path to NPZ data directory
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
    """
    print("="*80)
    print("Model Evaluation")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {npz_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = NPZWindowDataset(npz_dir, return_metadata=True)
    print(f"✓ Loaded dataset with {len(dataset)} windows")
    
    # Collate function for variable channels
    def collate_fn(batch):
        """Collate function that handles variable channels with padding."""
        signals_list, coords_list, regs_list, labels_list, metadata_list = zip(*batch)
        
        # Find max channels in this batch
        max_ch = max(s.shape[0] for s in signals_list)
        n_timesteps = signals_list[0].shape[1]
        batch_size = len(signals_list)
        
        # Pad to max channels
        signals_padded = torch.zeros(batch_size, max_ch, n_timesteps)
        coords_padded = torch.zeros(batch_size, max_ch, 3)
        regs_padded = torch.zeros(batch_size, max_ch, dtype=torch.long)
        
        for i, (sig, coord, reg) in enumerate(zip(signals_list, coords_list, regs_list)):
            n_ch = sig.shape[0]
            signals_padded[i, :n_ch] = sig
            coords_padded[i, :n_ch] = coord
            regs_padded[i, :n_ch] = reg
        
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        
        return signals_padded, coords_padded, regs_padded, labels_tensor, metadata_list
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Load model
    print("✓ Loading model from checkpoint...")
    model = IctalClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✓ Model loaded on {device}")
    
    # Run evaluation
    print("\n" + "="*80)
    print("Running evaluation...")
    print("="*80)
    
    all_probs = []
    all_preds = []
    all_labels = []
    all_patient_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            signals, coords, regs, labels, metadata = batch
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(signals)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(ictal)
            preds = torch.argmax(logits, dim=1)
            
            # Collect results
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_patient_ids.extend([m['patient_id'] for m in metadata])
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} samples")
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_patient_ids = np.array(all_patient_ids)
    
    print(f"\n✓ Evaluation complete: {len(all_labels)} samples")
    
    # Calculate metrics
    print("\n" + "="*80)
    print("Overall Metrics")
    print("="*80)
    
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
    
    accuracy = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUROC:     {auroc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'metric': ['accuracy', 'auroc', 'f1', 'precision', 'recall'],
        'value': [accuracy, auroc, f1, precision, recall]
    })
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'metrics.csv'}")
    
    # Confusion matrix
    print("\n" + "="*80)
    print("Confusion Matrix")
    print("="*80)
    
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Interictal', 'Ictal'],
                yticklabels=['Interictal', 'Ictal'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    plt.close()
    print(f"✓ Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    
    # Classification report
    print("\n" + "="*80)
    print("Classification Report")
    print("="*80)
    
    report = classification_report(all_labels, all_preds, 
                                   target_names=['Interictal', 'Ictal'],
                                   digits=4)
    print(report)
    
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    print(f"✓ Saved report to {output_dir / 'classification_report.txt'}")
    
    # ROC curve
    print("\n" + "="*80)
    print("ROC Curve")
    print("="*80)
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300)
    plt.close()
    print(f"✓ Saved ROC curve to {output_dir / 'roc_curve.png'}")
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300)
    plt.close()
    print(f"✓ Saved PR curve to {output_dir / 'precision_recall_curve.png'}")
    
    # Per-patient analysis
    print("\n" + "="*80)
    print("Per-Patient Analysis")
    print("="*80)
    
    patient_results = []
    for patient_id in np.unique(all_patient_ids):
        mask = all_patient_ids == patient_id
        patient_labels = all_labels[mask]
        patient_preds = all_preds[mask]
        patient_probs = all_probs[mask]
        
        if len(patient_labels) > 0:
            pat_accuracy = accuracy_score(patient_labels, patient_preds)
            pat_f1 = f1_score(patient_labels, patient_preds) if len(np.unique(patient_labels)) > 1 else 0.0
            
            patient_results.append({
                'patient_id': patient_id,
                'n_windows': len(patient_labels),
                'n_ictal': np.sum(patient_labels == 1),
                'n_interictal': np.sum(patient_labels == 0),
                'accuracy': pat_accuracy,
                'f1_score': pat_f1
            })
    
    patient_df = pd.DataFrame(patient_results)
    patient_df = patient_df.sort_values('f1_score', ascending=False)
    patient_df.to_csv(output_dir / 'per_patient_results.csv', index=False)
    
    print(f"\nTop 5 patients by F1 score:")
    print(patient_df.head(5).to_string())
    print(f"\n✓ Saved per-patient results to {output_dir / 'per_patient_results.csv'}")
    
    # Save all predictions
    predictions_df = pd.DataFrame({
        'patient_id': all_patient_ids,
        'true_label': all_labels,
        'pred_label': all_preds,
        'pred_prob': all_probs
    })
    predictions_df.to_csv(output_dir / 'all_predictions.csv', index=False)
    print(f"✓ Saved all predictions to {output_dir / 'all_predictions.csv'}")
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained benchmark models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--npz-dir',
        type=str,
        default='data/all_windows_per_patient',
        help='Path to NPZ data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        npz_dir=args.npz_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
