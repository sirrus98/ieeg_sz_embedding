"""
Model wrappers for channel-wise processing and aggregation.

These wrappers enable univariate models (ONSET, WVNT, Conformer) to process
multi-channel iEEG data by:
1. Reshaping channels as batch dimension
2. Processing each channel independently
3. Aggregating features across channels (mean or max pooling)
4. Final MLP classification
"""

from .onset_wrapper import ONSETChannelAggregator
from .wvnt_wrapper import WVNTChannelAggregator
from .conformer_wrapper import ConformerChannelAggregator
from .conformer_regions_wrapper import ConformerRegionsChannelAggregator
from .st_wrapper import STChannelAggregator
from .base_wrapper import BaseChannelAggregator

__all__ = [
    'BaseChannelAggregator',
    'ONSETChannelAggregator',
    'WVNTChannelAggregator',
    'ConformerChannelAggregator',
    'ConformerRegionsChannelAggregator',
    'STChannelAggregator',
]
