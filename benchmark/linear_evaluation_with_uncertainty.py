"""
Linear evaluation with per-fold metrics for uncertainty estimation.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import pytorch_lightning as pl


class LinearClassifier(pl.LightningModule):
    """2-layer MLP classifier."""
    
    def __init__(self, input_dim, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
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
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def run_cv_with_fold_metrics(X, y, n_splits=10, num_epochs=500, batch_size=32, 
                             learning_rate=1e-3, random_seed=42):
    """
    Run CV and return per-fold metrics for uncertainty estimation.
    
    Returns:
        fold_balanced_accs: List of balanced accuracies per fold
        fold_accs: List of accuracies per fold
        y_encoded: Encoded integer labels
        y_pred_all: Predictions for all samples
    """
    np.random.seed(random_seed)
    
    # Encode labels
    unique_labels = sorted(np.unique(y))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[label] for label in y])
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    
    # CV setup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    n_samples = X_tensor.shape[0]
    y_pred_all = np.empty(n_samples, dtype=int)
    
    fold_balanced_accs = []
    fold_accs = []
    
    print(f"\nRunning {n_splits}-fold cross-validation with per-fold metrics...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor)):
        print(f"--- Fold {fold+1}/{n_splits} ---", end=" ")
        
        # Create datasets
        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        test_dataset = TensorDataset(X_tensor[test_idx], y_tensor[test_idx])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        model = LinearClassifier(
            input_dim=X_tensor.shape[1],
            num_classes=len(unique_labels),
            learning_rate=learning_rate
        )
        
        # Train
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator="auto",
            devices=1
        )
        trainer.fit(model, train_loader)
        
        # Predict
        predictions = trainer.predict(model, dataloaders=test_loader)
        preds_fold = torch.cat(predictions).cpu().numpy()
        y_pred_all[test_idx] = preds_fold
        
        # Compute fold metrics
        y_true_fold = y_encoded[test_idx]
        fold_balanced_acc = balanced_accuracy_score(y_true_fold, preds_fold)
        fold_acc = accuracy_score(y_true_fold, preds_fold)
        
        fold_balanced_accs.append(fold_balanced_acc)
        fold_accs.append(fold_acc)
        
        print(f"Bal_Acc: {fold_balanced_acc:.4f}, Acc: {fold_acc:.4f}")
    
    return fold_balanced_accs, fold_accs, y_encoded, y_pred_all, unique_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings_path', type=str)
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()
    
    model_name = Path(args.embeddings_path).stem
    
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name.upper()} with per-fold metrics")
    print('='*80)
    
    # Load embeddings
    data = np.load(args.embeddings_path, allow_pickle=True)
    X = data['embeddings']
    y = data['labels']
    
    print(f"  Embeddings shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    
    # Run CV with fold metrics
    fold_balanced_accs, fold_accs, y_encoded, y_pred_all, unique_labels = run_cv_with_fold_metrics(
        X, y, n_splits=10, num_epochs=args.epochs
    )
    
    # Compute overall metrics
    overall_balanced_acc = balanced_accuracy_score(y_encoded, y_pred_all)
    overall_acc = accuracy_score(y_encoded, y_pred_all)
    
    # Compute uncertainties (std from folds)
    balanced_acc_std = np.std(fold_balanced_accs)
    acc_std = np.std(fold_accs)
    
    # Display results
    print(f"\n{'='*80}")
    print("RESULTS WITH UNCERTAINTY")
    print('='*80)
    print(f"Balanced Accuracy: {overall_balanced_acc:.4f} ± {balanced_acc_std:.4f}")
    print(f"Accuracy:          {overall_acc:.4f} ± {acc_std:.4f}")
    print()
    print("Per-fold Balanced Accuracies:")
    for i, ba in enumerate(fold_balanced_accs, 1):
        print(f"  Fold {i:2d}: {ba:.4f}")
    print()
    print("Per-fold Accuracies:")
    for i, a in enumerate(fold_accs, 1):
        print(f"  Fold {i:2d}: {a:.4f}")
    print('='*80)


if __name__ == '__main__':
    main()
