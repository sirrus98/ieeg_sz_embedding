"""
Base class for channel aggregation wrappers.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseChannelAggregator(nn.Module, ABC):
    """
    Abstract base class for channel aggregation wrappers.
    
    Common pattern:
    1. Reshape [batch, n_channels, seq_len] → [batch*n_channels, 1, seq_len]
    2. Extract features from univariate model
    3. Reshape [batch*n_channels, feat_dim] → [batch, n_channels, feat_dim]
    4. Aggregate channels (mean or max pooling)
    5. MLP classifier → [batch, n_classes]
    """
    
    def __init__(self, aggregation='mean'):
        """
        Args:
            aggregation: 'mean' or 'max' pooling across channels
        """
        super().__init__()
        assert aggregation in ['mean', 'max'], "aggregation must be 'mean' or 'max'"
        self.aggregation = aggregation
    
    @abstractmethod
    def extract_features(self, x):
        """
        Extract features from univariate model.
        
        Args:
            x: [batch*n_channels, 1, seq_len]
        
        Returns:
            features: [batch*n_channels, feat_dim]
        """
        pass
    
    def aggregate_channels(self, features, batch_size, n_channels):
        """
        Aggregate features across channels.
        
        Args:
            features: [batch*n_channels, feat_dim]
            batch_size: original batch size
            n_channels: number of channels per sample
        
        Returns:
            aggregated: [batch, feat_dim]
        """
        # Reshape to [batch, n_channels, feat_dim]
        features = features.view(batch_size, n_channels, -1)
        
        # Aggregate
        if self.aggregation == 'mean':
            aggregated = features.mean(dim=1)  # [batch, feat_dim]
        elif self.aggregation == 'max':
            aggregated = features.max(dim=1)[0]  # [batch, feat_dim]
        
        return aggregated
    
    def forward(self, x):
        """
        Forward pass with channel aggregation.
        
        Args:
            x: [batch, n_channels, seq_len] - multi-channel time series
        
        Returns:
            logits: [batch, n_classes] - classification logits
        """
        batch_size, n_channels, seq_len = x.shape
        
        # Reshape: channels → batch
        x = x.view(batch_size * n_channels, 1, seq_len)
        
        # Extract features
        features = self.extract_features(x)  # [batch*n_channels, feat_dim]
        
        # Aggregate channels
        aggregated = self.aggregate_channels(features, batch_size, n_channels)
        
        # Classify
        logits = self.classifier(aggregated)
        
        return logits
