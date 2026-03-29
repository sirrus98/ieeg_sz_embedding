# %%
import os
import pickle

import torch
import numpy as np

from config import CONFIG
from model_config import MODEL_CONFIG, ModelConfig, get_model_config

import torchvision.transforms as T
from torch.utils.data import Dataset

TARGET_SF = 256

# make sz_table
sz_table = CONFIG.sz_times.copy()
sz_table.index = sz_table.index.map(CONFIG.hup_to_rid)
sz_table.reset_index(inplace=True)


def get_data_path(duration: str, data_dir: str = CONFIG.data_dir) -> str:
    """Return the expected pkl path for a given recording duration.

    Args:
        duration: One of ``"10s"``, ``"20s"``, ``"20s_peri"``, ``"45s"``,
            or ``"full"``.
        data_dir: Base directory containing the pkl files.
    """
    return os.path.join(
        data_dir,
        f"all_sz_coords_data_spatiotemporal_regs_{duration}.pkl",
    )


class SzDataset(Dataset):
    """General-purpose iEEG seizure dataset.

    Loads a pickle produced by ``make_dataset.py`` and exposes multi-crop
    views for SwAV self-supervised training.  Crop sizes are driven by
    ``model_config.multi_crop_config``.

    For fixed-duration datasets (10s/20s/45s) all signals share the same
    time length, so they are pre-stacked into a contiguous tensor.
    For variable-length datasets (``full``), signals are kept as a Python
    list and each item's actual time length is tracked so that crops are
    never drawn from zero-padded regions.

    Args:
        data_path:    Full path to the ``.pkl`` file.
        model_config: :class:`~model_config.ModelConfig` instance.
        subsample:    Fraction of channels to keep (1 = all).
        transform:    Apply colour-jitter / blur augmentation to crops.
        random_seed:  Seed for reproducible channel sub-sampling.
        shuffle:      Randomise channel order (requires ``subsample < 1``).
    """

    def __init__(
        self,
        data_path: str,
        model_config: ModelConfig,
        subsample: float = 1,
        transform: bool = False,
        random_seed: int = 42,
        shuffle: bool = False,
    ):
        self.model_config = model_config

        with open(data_path, "rb") as f:
            loaded = pickle.load(f)

        # pkl format: (signals, coords, regs, ch_names, indices)
        # older format omits ch_names: (signals, coords, regs, indices)
        if len(loaded) == 5:
            signals, coords, regs, _ch_names, indices = loaded
        else:
            signals, coords, regs, indices = loaded

        new_size = int(model_config.channel_buffer_size * subsample)

        if subsample != 1 or shuffle:
            new_signals, new_coords, new_regs = [], [], []
            rng = torch.Generator().manual_seed(random_seed)
            for sig, coord, reg in zip(signals, coords, regs):
                n_ch = sig.shape[0]
                selected_idx = torch.randint(
                    low=0,
                    high=n_ch,
                    size=(int(n_ch * subsample),),
                    generator=rng,
                )
                selected_idx, _ = torch.sort(selected_idx)
                sig = sig[selected_idx]
                coord = coord[selected_idx]
                reg = reg[selected_idx]
                new_signals.append(
                    torch.nn.functional.pad(
                        torch.from_numpy(sig), (0, 0, 0, new_size - sig.shape[0])
                    )
                )
                new_coords.append(
                    torch.nn.functional.pad(
                        torch.from_numpy(coord), (0, 0, 0, new_size - coord.shape[0])
                    )
                )
                new_regs.append(
                    torch.nn.functional.pad(
                        torch.from_numpy(reg) + 1, (0, new_size - reg.shape[0])
                    )
                )
            padded_signals = new_signals
            padded_coords = new_coords
            padded_regs = new_regs
        else:
            padded_signals = [
                torch.nn.functional.pad(
                    torch.from_numpy(sig), (0, 0, 0, new_size - sig.shape[0])
                )
                for sig in signals
            ]
            padded_coords = [
                torch.nn.functional.pad(
                    torch.from_numpy(coord), (0, 0, 0, new_size - coord.shape[0])
                )
                for coord in coords
            ]
            padded_regs = [
                torch.nn.functional.pad(
                    torch.from_numpy(reg) + 1, (0, new_size - reg.shape[0])
                )
                for reg in regs
            ]

        # Track actual time lengths so crops don't stray into zero-padded regions.
        # Channel padding is on dim 0 (already handled above); time lengths live
        # on dim 1 and may vary for "full" duration recordings.
        self.actual_lengths = [s.shape[-1] for s in padded_signals]
        time_lengths_set = set(self.actual_lengths)
        self._variable_length = len(time_lengths_set) > 1

        if self._variable_length:
            # Keep as list; stacking would fail due to unequal time dims.
            self.signals = padded_signals
            self.coords = padded_coords
            self.regs = padded_regs
        else:
            self.signals = torch.stack(padded_signals, dim=0)
            self.coords = torch.stack(padded_coords, dim=0)
            self.regs = torch.stack(padded_regs, dim=0)

        metadata = sz_table.loc[indices]
        assert len(metadata) == len(signals)
        metadata.reset_index(inplace=True)
        self.metadata = metadata
        self.transform = transform
        self.random_seed = random_seed
        self.subsample = subsample

        self.tensor_transform = T.Compose(
            [
                T.RandomApply(
                    [T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)],
                    p=0.8,
                ),
                T.RandomApply([T.GaussianBlur(kernel_size=(5, 5))], p=0.6),
            ]
        )

    # ------------------------------------------------------------------
    def get_cropped_data(
        self, tensor: torch.Tensor, cropped_len: int, actual_length: int = None
    ) -> torch.Tensor:
        """Return a random contiguous crop of *cropped_len* time steps.

        Args:
            tensor:        Signal tensor of shape ``(channels, time)``.
            cropped_len:   Number of time steps in the crop.
            actual_length: Real data length (ignores any time-axis padding).
                           Defaults to the full tensor width.
        """
        limit = actual_length if actual_length is not None else tensor.size(-1)
        high = limit - cropped_len
        if high <= 0:
            # Signal shorter than requested crop — return from the start
            return tensor[..., :cropped_len]
        start = torch.randint(low=0, high=high, size=(1,)).item()
        idx = (start + torch.arange(cropped_len)).unsqueeze(0).expand(
            self.model_config.channel_buffer_size, -1
        )
        return tensor.gather(dim=1, index=idx)

    def multicrop(self, tensor: torch.Tensor, actual_length: int = None) -> list:
        """Return one random crop per size in ``model_config.multi_crop_config``."""
        size_list = self.model_config.multi_crop_config
        if self.transform:
            return [
                self.tensor_transform(
                    self.get_cropped_data(tensor, sl, actual_length).unsqueeze(0)
                ).squeeze(0)
                for sl in size_list
            ]
        return [self.get_cropped_data(tensor, sl, actual_length) for sl in size_list]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        actual_len = self.actual_lengths[idx] if self._variable_length else None
        crops = self.multicrop(signal, actual_len)
        return crops, self.regs[idx], idx


# ---------------------------------------------------------------------------
# Legacy / backward-compatible classes (unchanged public API)
# ---------------------------------------------------------------------------

class SzDatasetRegs(Dataset):
    """Legacy dataset — loads the original (non-versioned) pkl."""

    def __init__(self, transform: bool = False):
        fname = os.path.join(CONFIG.data_dir, "all_sz_coords_data_spatiotemporal_regs.pkl")
        with open(fname, "rb") as f:
            signals, coords, regs, indices = pickle.load(f)

        new_size = MODEL_CONFIG.channel_buffer_size
        self.signals = torch.stack(
            [torch.nn.functional.pad(torch.from_numpy(sig), (0, 0, 0, new_size - sig.shape[0])) for sig in signals], dim=0
        )
        self.coords = torch.stack(
            [torch.nn.functional.pad(torch.from_numpy(coord), (0, 0, 0, new_size - coord.shape[0])) for coord in coords], dim=0
        )
        self.regs = torch.stack(
            [torch.nn.functional.pad(torch.from_numpy(reg) + 1, (0, new_size - reg.shape[0])) for reg in regs], dim=0
        )

        metadata = sz_table.loc[indices]
        assert len(metadata) == len(signals)
        metadata.reset_index(inplace=True)
        self.metadata = metadata
        self.transform = transform

        self.tensor_transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=(5, 5))], p=0.6),
        ])

    def get_cropped_data(self, tensor, cropped_len):
        choices = torch.randint(low=0, high=tensor.size()[-1] - cropped_len, size=(1,))
        indices = (choices + torch.arange(cropped_len)).repeat(MODEL_CONFIG.channel_buffer_size, 1)
        return tensor.gather(dim=1, index=indices)

    def multicrop(self, tensor):
        size_list = MODEL_CONFIG.multi_crop_config
        if self.transform:
            return [self.tensor_transform(self.get_cropped_data(tensor, sl).unsqueeze(0)).squeeze(0) for sl in size_list]
        return [self.get_cropped_data(tensor, sl) for sl in size_list]

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.multicrop(self.signals[idx]), self.regs[idx], idx


class SzDatasetRegsFull(Dataset):
    """Legacy dataset — loads the 10 s pkl and exposes raw (non-cropped) items."""

    def __init__(self, transform: bool = False):
        fname = os.path.join(CONFIG.data_dir, "all_sz_coords_data_spatiotemporal_regs_10s.pkl")
        with open(fname, "rb") as f:
            signals, coords, regs, ch_names, indices = pickle.load(f)

        new_size = MODEL_CONFIG.channel_buffer_size
        self.signals = torch.stack(
            [torch.nn.functional.pad(torch.from_numpy(sig), (0, 0, 0, new_size - sig.shape[0])) for sig in signals], dim=0
        )
        self.coords = torch.stack(
            [torch.nn.functional.pad(torch.from_numpy(coord), (0, 0, 0, new_size - coord.shape[0])) for coord in coords], dim=0
        )
        self.regs = torch.stack(
            [torch.nn.functional.pad(torch.from_numpy(reg) + 1, (0, new_size - reg.shape[0])) for reg in regs], dim=0
        )

        metadata = sz_table.loc[indices]
        assert len(metadata) == len(signals)
        metadata.reset_index(inplace=True)
        self.metadata = metadata
        self.ch_names = ch_names

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.regs[idx], [str(ch) for ch in self.ch_names[idx]], idx


class SzDatasetRegsTenS(SzDataset):
    """Backward-compatible 10 s training dataset."""

    def __init__(self, subsample: float = 1, transform: bool = False,
                 random_seed: int = 42, shuffle: bool = False):
        super().__init__(
            data_path=get_data_path("10s"),
            model_config=get_model_config("10s"),
            subsample=subsample,
            transform=transform,
            random_seed=random_seed,
            shuffle=shuffle,
        )


# %%
if __name__ == "__main__":
    dataset = SzDatasetRegsTenS()
