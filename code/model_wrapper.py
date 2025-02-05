# %%
# imports
# autoreload
# %load_ext autoreload
# %autoreload 2
import torch
import torch.nn as nn
from torch import Tensor
import math
from torchsummary import summary
from dataset import SzDatasetRegs
import lightning as L
import torch.optim as optim
import torch.nn.functional as F
from lightly.loss import SwaVLoss
from lightly.models import utils
from typing import Optional, Sequence, Tuple, Union
from model_config import MODEL_CONFIG
from torch.nn import Module


class MemoryBankModule(Module):
    """Memory bank implementation

    This is a parent class to all loss functions implemented by the lightly
    Python package. This way, any loss can be used with a memory bank if
    desired.

    Attributes:
        size:
            Size of the memory bank as (num_features, dim) tuple. If num_features is 0
            then the memory bank is disabled. Deprecated: If only a single integer is
            passed, it is interpreted as the number of features and the feature
            dimension is inferred from the first batch stored in the memory bank.
            Leaving out the feature dimension might lead to errors in distributed
            training.
        gather_distributed:
            If True then negatives from all gpus are gathered before the memory bank
            is updated. This results in more frequent updates of the memory bank and
            keeps the memory bank contents independent of the number of gpus. But it has
            the drawback that synchronization between processes is required and
            diversity of the memory bank content is reduced.
        feature_dim_first:
            If True, the memory bank returns features with shape (dim, num_features).
            If False, the memory bank returns features with shape (num_features, dim).

    Examples:
        >>> class MyLossFunction(MemoryBankModule):
        >>>
        >>>     def __init__(self, memory_bank_size: Tuple[int, int] = (2 ** 16, 128)):
        >>>         super().__init__(memory_bank_size)
        >>>
        >>>     def forward(self, output: Tensor, labels: Optional[Tensor] = None):
        >>>         output, negatives = super().forward(output)
        >>>
        >>>         if negatives is not None:
        >>>             # evaluate loss with negative samples
        >>>         else:
        >>>             # evaluate loss without negative samples

    """

    def __init__(
            self,
            size: Union[int, Sequence[int]] = 65536,
            gather_distributed: bool = False,
            feature_dim_first: bool = True,
    ):
        super().__init__()
        size_tuple = (size,) if isinstance(size, int) else tuple(size)

        if any(x < 0 for x in size_tuple):
            raise ValueError(
                f"Illegal memory bank size {size}, all entries must be non-negative."
            )

        self.size = size_tuple
        self.gather_distributed = gather_distributed
        self.feature_dim_first = feature_dim_first
        self.bank: Tensor
        self.register_buffer(
            "bank",
            tensor=torch.empty(size=size_tuple, dtype=torch.float),
            persistent=False,
        )
        self.bank_ptr: Tensor
        self.register_buffer(
            "bank_ptr",
            tensor=torch.empty(1, dtype=torch.long),
            persistent=False,
        )

        if isinstance(size, int) and size > 0:
            warnings.warn(
                (
                    f"Memory bank size 'size={size}' does not specify feature "
                    "dimension. It is recommended to set the feature dimension with "
                    "'size=(n, dim)' when creating the memory bank. Distributed "
                    "training might fail if the feature dimension is not set."
                ),
                UserWarning,
            )
        elif len(size_tuple) > 1:
            self._init_memory_bank(size=size_tuple)

    @torch.no_grad()
    def _init_memory_bank(self, size: Tuple[int, ...]) -> None:
        """Initialize the memory bank.

        Args:
            size:
                Size of the memory bank as (num_features, dim) tuple.

        """
        self.bank = torch.randn(size).type_as(self.bank)
        self.bank = torch.nn.functional.normalize(self.bank, dim=-1)
        self.bank_ptr = torch.zeros(1).type_as(self.bank_ptr)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: Tensor) -> None:
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """
        if self.gather_distributed:
            batch = utils.concat_all_gather(batch)

        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)
        if ptr + batch_size >= self.size[0]:
            self.bank[ptr:] = batch[: self.size[0] - ptr].detach()
            self.bank_ptr.zero_()
        else:
            self.bank[ptr: ptr + batch_size] = batch.detach()
            self.bank_ptr[0] = ptr + batch_size

    def forward(
            self,
            output: Tensor,
            labels: Optional[Tensor] = None,
            update: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None]]:
        """Query memory bank for additional negative samples

        Args:
            output:
                The output of the model.
            labels:
                Should always be None, will be ignored.
            update:
                If True, the memory bank will be updated with the current output.

        Returns:
            The output if the memory bank is of size 0, otherwise the output
            and the entries from the memory bank. Entries from the memory bank have
            shape (dim, num_features) if feature_dim_first is True and
            (num_features, dim) otherwise.

        """

        # no memory bank, return the output
        if self.size[0] == 0:
            return output, None

        # Initialize the memory bank if it is not already done.
        if self.bank.ndim == 1:
            dim = output.shape[1:]
            self._init_memory_bank(size=(*self.size, *dim))

        # query and update memory bank
        bank = self.bank.clone().detach()
        if self.feature_dim_first:
            # swap bank size and feature dimension for backwards compatibility
            bank = bank.transpose(0, -1)

        # only update memory bank if we later do backward pass (gradient)
        if update:
            self._dequeue_and_enqueue(output)

        return output, bank


def create_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0


def create_shuffled_positive_target(input_segments, chunk_size, fs=256):
    """
    Create a target tensor for time-contrastive loss with shuffled positive segments.

    Parameters:
    - input_segments: Tensor representing input EEG segments with shape (batch_size, signals, time_samples).
    - chunk_size: Size of each chunk to maintain intact while shuffling in seconds.
    - fs: Sampling frequency of the EEG segments in Hz.

    Returns:
    - target: Tensor representing the shuffled positive segments.
    """
    batch_size, signals, time_samples = input_segments.size()

    # Calculate the number of chunks
    chunk_size_samples = int(chunk_size * fs)
    num_chunks = time_samples // chunk_size_samples

    # Reshape the input_segments tensor into chunks
    reshaped_segments = input_segments.view(batch_size, signals, num_chunks, chunk_size_samples)

    # Create indices for shuffling chunks
    shuffled_chunk_indices = torch.randperm(num_chunks)

    # Shuffle the chunks
    shuffled_segments = reshaped_segments[:, :, shuffled_chunk_indices, :]

    # Reshape back to the original shape
    target = shuffled_segments.view(batch_size, signals, time_samples)

    return target


# %%
class ModelWrapper(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = NTXentLoss()

    def training_step(self, batch, batch_idx):
        signals, coords, idx = batch

        signals = signals.float()
        coords = coords.float()

        # set nans to 0
        signals[signals.isnan()] = 0

        # mask = np.any((signals.cpu().numpy() != 0), axis=-1)

        # forward pass
        output = self.model(signals, coords)

        # shuffled signals for contrastive loss
        shuffled_signals = create_shuffled_positive_target(signals, 5, fs=256)
        shuffled_output = self.model(shuffled_signals, coords)

        # normalize both
        output = F.normalize(output, p=2, dim=-1)
        shuffled_output = F.normalize(shuffled_output, p=2, dim=-1)

        # flatten output 1 and 2 dimensions (batch_size, seq_len, hidden_size) -> (batch_size, seq_len * hidden_size)
        out_flat = output.flatten(1, 2)
        shuffled_output_flat = shuffled_output.flatten(1, 2)

        loss = self.criterion(out_flat, shuffled_output_flat)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=MODEL_CONFIG.learning_rate)
        # early stopping
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
            'monitor': 'train_loss'
        }


class ModelV3Wrapper(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.criterion = DCLLoss()
        self.criterion = SwaVLoss()

    def training_step(self, batch, batch_idx):
        # normalize prototype so it is on a sphere
        self.model.prototypes.normalize()
        signals, coords, idx = batch

        coords = coords.float()

        # pass multicrop views to the model
        multi_crop_features = [self.model.prototypes(
            self.model(torch.nan_to_num(view.float(), posinf=0.0, neginf=0.0).to(self.device), coords),
            self.current_epoch) for view in signals]

        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]

        loss = self.criterion(high_resolution, low_resolution)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=MODEL_CONFIG.learning_rate)
        # early stopping
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
        #     'monitor': 'train_loss'
        # }

        return optimizer


class ModelV3WrapperWithQueue(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = SwaVLoss()

        self.start_queue_at_epoch = 50
        self.queues = nn.ModuleList([MemoryBankModule(size=(256, 128)) for _ in range(2)])

    def training_step(self, batch, batch_idx):
        # normalize prototype so it is on a sphere
        self.model.prototypes.normalize()
        signals, regs, idx = batch

        regs = regs.long()

        high_resolution, low_resolution = signals[:2], signals[2:]

        high_resolution_features = [self.model(torch.nan_to_num(x.float(), posinf=0.0, neginf=0.0).to(self.device)
                                               , regs) for x in high_resolution]
        low_resolution_features = [self.model(torch.nan_to_num(x.float(), posinf=0.0, neginf=0.0).to(self.device)
                                              , regs) for x in low_resolution]

        high_resolution_prototypes = [self.model.prototypes(x, self.current_epoch) for x in high_resolution_features]
        low_resolution_prototypes = [self.model.prototypes(x, self.current_epoch) for x in low_resolution_features]

        queue_prototypes = self._get_queue_prototypes(high_resolution_features)

        loss = self.criterion(high_resolution_prototypes, low_resolution_prototypes, queue_prototypes)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    @torch.no_grad()
    def _get_queue_prototypes(self, high_resolution_features):
        if len(high_resolution_features) != len(self.queues):
            raise ValueError(
                f"The number of queues ({len(self.queues)}) should be equal to the number of high "
                f"resolution inputs ({len(high_resolution_features)}). Set `n_queues` accordingly."
            )

        # Get the queue features
        queue_features = []
        for i in range(len(self.queues)):
            _, features = self.queues[i](high_resolution_features[i], update=True)
            # Queue features are in (num_ftrs X queue_length) shape, while the high res
            # features are in (batch_size X num_ftrs). Swap the axes for interoperability.
            features = torch.permute(features, (1, 0))
            queue_features.append(features)

        # If loss calculation with queue prototypes starts at a later epoch,
        # just queue the features and return None instead of queue prototypes.
        if (
                self.start_queue_at_epoch > 0
                and self.current_epoch < self.start_queue_at_epoch
        ):
            return None

        # Assign prototypes
        queue_prototypes = [
            self.model.prototypes(x, self.current_epoch) for x in queue_features
        ]
        return queue_prototypes

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=MODEL_CONFIG.learning_rate)
        # early stopping
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
        #     'monitor': 'train_loss'
        # }

        return optimizer
