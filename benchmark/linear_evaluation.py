"""
Linear evaluation of embeddings for onset location classification.

This script replicates the linear evaluation experiment from the notebook (cells 101-103),
using 10-fold cross-validation with a 2-layer MLP classifier.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (
    balanced_accuracy_score, 
    accuracy_score,
    confusion_matrix, 
    classification_report
)
import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt


class LinearClassifier(pl.LightningModule):
    """
    2-layer MLP classifier for linear evaluation.
    
    Architecture from notebook cell 101:
    - Linear layer 1: input_dim -> input_dim
    - Linear layer 2: input_dim -> num_classes
    """
    
    def __init__(self, input_dim, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  # saves hyperparameters to self.hparams
        self.linear_1 = nn.Linear(input_dim, input_dim)
        self.linear = nn.Linear(input_dim, num_classes)
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.linear_1(x)
        return self.linear(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log("test_loss", loss, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": y}
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def load_embeddings(npz_path):
    """
    Load embeddings and labels from npz file.
    
    Returns:
        X: [n_samples, embed_dim] numpy array
        y: [n_samples] numpy array of label strings
        patient_ids: [n_samples] numpy array of patient IDs
    """
    print(f"Loading embeddings from {npz_path}...")
    
    data = np.load(npz_path, allow_pickle=True)
    
    X = data['embeddings']
    y = data['labels']
    patient_ids = data['patient_ids']
    
    print(f"  ✓ Embeddings shape: {X.shape}")
    print(f"  ✓ Labels shape: {y.shape}")
    print(f"  ✓ Unique labels: {np.unique(y)}")
    print(f"  ✓ Label counts:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    {label}: {count}")
    
    return X, y, patient_ids


def run_cross_validation(X, y, n_splits=10, num_epochs=500, batch_size=32, 
                        learning_rate=1e-3, random_seed=42):
    """
    Run 10-fold cross-validation with PyTorch Lightning.
    
    Args:
        X: [n_samples, embed_dim] features
        y: [n_samples] labels (strings)
        n_splits: Number of CV folds
        num_epochs: Training epochs per fold
        batch_size: Batch size
        learning_rate: Learning rate
        random_seed: Random seed for reproducibility
    
    Returns:
        y_encoded: Encoded integer labels
        y_pred_all: Predictions for all samples
        unique_labels: Sorted list of unique label strings
    """
    np.random.seed(random_seed)
    
    # Encode string labels to integers
    unique_labels = sorted(np.unique(y))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[label] for label in y])
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    
    # Prepare for cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    n_samples = X_tensor.shape[0]
    y_pred_all = np.empty(n_samples, dtype=int)
    
    print(f"\nRunning {n_splits}-fold cross-validation...")
    print(f"  Training epochs per fold: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print()
    
    # Cross-validation loop
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor)):
        print(f"--- Fold {fold+1}/{n_splits} ---")
        
        # Create TensorDatasets and DataLoaders for the current fold
        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        test_dataset = TensorDataset(X_tensor[test_idx], y_tensor[test_idx])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize the model
        model = LinearClassifier(
            input_dim=X_tensor.shape[1],
            num_classes=len(unique_labels),
            learning_rate=learning_rate
        )
        
        # Initialize the PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            logger=False,  # disable logging for simplicity
            enable_checkpointing=False,  # disable checkpointing
            enable_progress_bar=False,  # disable progress bar
            accelerator="auto",
            devices=1
        )
        
        # Train the model on the training split of the current fold
        trainer.fit(model, train_loader)
        
        # Use the trainer to predict on the test split
        predictions = trainer.predict(model, dataloaders=test_loader)
        # 'predictions' is a list of tensors (one per batch); concatenate them:
        preds_fold = torch.cat(predictions).cpu().numpy()
        
        # Store the predictions in their corresponding indices from the original dataset
        y_pred_all[test_idx] = preds_fold
        
        print(f"  ✓ Fold {fold+1} complete")
    
    return y_encoded, y_pred_all, unique_labels


def compute_metrics(y_true, y_pred, unique_labels):
    """
    Compute and display metrics.
    
    Args:
        y_true: True labels (encoded as integers)
        y_pred: Predicted labels (encoded as integers)
        unique_labels: List of label names
    
    Returns:
        metrics: Dictionary of computed metrics
    """
    # Compute metrics
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print('='*80)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print()
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=unique_labels)
    print("Classification Report:")
    print(report)
    
    # Confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(conf_mat)
    
    return {
        'balanced_accuracy': balanced_acc,
        'accuracy': acc,
        'confusion_matrix': conf_mat,
        'classification_report': report
    }


def plot_confusion_matrix(conf_mat, unique_labels, output_path):
    """
    Plot and save confusion matrix.
    
    Args:
        conf_mat: Confusion matrix array
        unique_labels: List of label names
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(
        conf_mat,
        ax=ax,
        annot=True,
        cmap='Blues',
        cbar=False,
        square=True,
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        fmt='d'
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {output_path}")
    plt.close()


def evaluate_embeddings(npz_path, output_dir=None, n_splits=10, num_epochs=500, 
                       batch_size=32, learning_rate=1e-3):
    """
    Main evaluation function.
    
    Args:
        npz_path: Path to embeddings npz file
        output_dir: Directory to save results (optional)
        n_splits: Number of CV folds
        num_epochs: Training epochs per fold
        batch_size: Batch size
        learning_rate: Learning rate
    """
    model_name = Path(npz_path).stem  # e.g., 'wvnt_mean_embeddings'
    
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name.upper()}")
    print('='*80)
    
    # Load embeddings
    X, y, patient_ids = load_embeddings(npz_path)
    
    # Run cross-validation
    y_encoded, y_pred_all, unique_labels = run_cross_validation(
        X, y, n_splits=n_splits, num_epochs=num_epochs, 
        batch_size=batch_size, learning_rate=learning_rate
    )
    
    # Compute metrics
    metrics = compute_metrics(y_encoded, y_pred_all, unique_labels)
    
    # Save results if output directory specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save confusion matrix plot
        cm_path = output_dir / f'{model_name}_confusion_matrix.png'
        plot_confusion_matrix(metrics['confusion_matrix'], unique_labels, cm_path)
        
        # Save metrics to text file
        metrics_path = output_dir / f'{model_name}_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"\nClassification Report:\n")
            f.write(metrics['classification_report'])
            f.write(f"\nConfusion Matrix:\n")
            f.write(str(metrics['confusion_matrix']))
        
        print(f"✓ Metrics saved to {metrics_path}")
    
    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Linear evaluation of seizure embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'embeddings_path',
        type=str,
        help='Path to embeddings npz file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save results (optional)'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=10,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='Training epochs per fold'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_embeddings(
        args.embeddings_path,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == '__main__':
    main()
