"""
PyTorch Lightning module for ictal/interictal classification.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score, Precision, Recall
from typing import Optional


class IctalClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for ictal/interictal classification.
    Wraps any of the baseline models (ONSET, WVNT, Conformer).
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        aggregation: str = 'mean',
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            model: Wrapper model (ONSETChannelAggregator, WVNTChannelAggregator, etc.)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for AdamW
            aggregation: Aggregation method ('mean' or 'max')
            class_weights: Class weights for imbalanced data (optional)
        """
        super().__init__()
        
        # Save hyperparameters (automatically logged to W&B/TensorBoard/CSV)
        self.save_hyperparameters(ignore=['model', 'class_weights'])
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Metrics for training
        self.train_acc = Accuracy(task='binary')
        
        # Metrics for validation
        self.val_acc = Accuracy(task='binary')
        self.val_auroc = AUROC(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')
        
        # Metrics for test
        self.test_acc = Accuracy(task='binary')
        self.test_auroc = AUROC(task='binary')
        self.test_f1 = F1Score(task='binary')
        self.test_precision = Precision(task='binary')
        self.test_recall = Recall(task='binary')
    
    def forward(self, x, regs=None):
        """
        Forward pass.
        
        Args:
            x: [batch, n_channels, 2560] - multi-channel time series
            regs: [batch, n_channels] - region IDs (optional, required for some models)
        
        Returns:
            logits: [batch, 2] - classification logits
        """
        # Check if model needs regions (ST model, ConformerRegions)
        needs_regions = (
            hasattr(self.model, 'st_model') or 
            hasattr(self.model, 'conformer_regions') or
            'Regions' in self.model.__class__.__name__ or
            'ST' in self.model.__class__.__name__
        )
        
        if needs_regions:
            return self.model(x, regs)
        else:
            return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: (signals, coords, regs, labels)
            batch_idx: Batch index
        
        Returns:
            loss: Training loss
        """
        signals, coords, regs, labels = batch
        # signals: [batch, max_channels, 2560]
        # regs: [batch, max_channels]
        # labels: [batch]
        
        # Pass regions if model needs them (ST model, ConformerRegions)
        needs_regions = (
            hasattr(self.model, 'st_model') or 
            hasattr(self.model, 'conformer_regions') or
            'Regions' in self.model.__class__.__name__ or
            'ST' in self.model.__class__.__name__
        )
        
        if needs_regions:
            logits = self(signals, regs)
        else:
            logits = self(signals)
        
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: (signals, coords, regs, labels)
            batch_idx: Batch index
        
        Returns:
            loss: Validation loss
        """
        signals, coords, regs, labels = batch
        
        # Pass regions if model needs them (ST model, ConformerRegions)
        # Check for wrapped model attributes
        needs_regions = (
            hasattr(self.model, 'st_model') or 
            hasattr(self.model, 'conformer_regions') or
            'Regions' in self.model.__class__.__name__ or
            'ST' in self.model.__class__.__name__
        )
        
        if needs_regions:
            logits = self(signals, regs)
        else:
            logits = self(signals)
        
        loss = self.criterion(logits, labels)
        
        probs = F.softmax(logits, dim=1)[:, 1]  # P(ictal)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.val_acc(preds, labels)
        self.val_auroc(probs, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        
        # Log metrics (on_epoch=True is critical for wandb logging!)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/auroc', self.val_auroc, on_step=False, on_epoch=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True)
        self.log('val/precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val/recall', self.val_recall, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: (signals, coords, regs, labels)
            batch_idx: Batch index
        
        Returns:
            dict: Predictions and labels
        """
        signals, coords, regs, labels = batch
        
        # Pass regions if model needs them (ST model, ConformerRegions)
        # Check for wrapped model attributes
        needs_regions = (
            hasattr(self.model, 'st_model') or 
            hasattr(self.model, 'conformer_regions') or
            'Regions' in self.model.__class__.__name__ or
            'ST' in self.model.__class__.__name__
        )
        
        if needs_regions:
            logits = self(signals, regs)
        else:
            logits = self(signals)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.test_acc(preds, labels)
        self.test_auroc(probs, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        
        # Log metrics (on_epoch=True is critical for wandb logging!)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test/auroc', self.test_auroc, on_step=False, on_epoch=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test/precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test/recall', self.test_recall, on_step=False, on_epoch=True)
        
        return {'probs': probs, 'preds': preds, 'labels': labels}
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            dict: Optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/auroc',
                'interval': 'epoch'
            }
        }
