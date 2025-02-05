# %%
import os
import pickle

import torch
import numpy as np

from config import CONFIG
from model_config import MODEL_CONFIG

import torchvision.transforms as T
from torch.utils.data import Dataset

TARGET_SF = 256

# make sz_table
sz_table = CONFIG.sz_times.copy()
sz_table.index = sz_table.index.map(CONFIG.hup_to_rid)
sz_table.reset_index(inplace=True)

class SzDatasetRegs(Dataset):
    def __init__(self, transform=False):
        """
        Initializes the Dataset object.

        Args:
            transform (bool): Flag indicating whether to apply data transformations. Default is False.
        """
        # load data
        fname = os.path.join(CONFIG.data_dir, "all_sz_coords_data_spatiotemporal_regs.pkl")
        with open(fname, 'rb') as f:
            signals, coords, regs, indices = pickle.load(f)

        # use torch padding to get the first dimension of signals and coords to be MODEL_CONFIG.channel_buffer_size
        new_size = MODEL_CONFIG.channel_buffer_size
        # pad signals
        signals = [torch.nn.functional.pad(torch.from_numpy(sig), (0, 0, 0, new_size - sig.shape[0])) for sig in signals]
        self.signals = torch.stack(signals, dim=0)

        # pad coords
        coords = [torch.nn.functional.pad(torch.from_numpy(coord), (0, 0, 0, new_size - coord.shape[0])) for coord in coords]
        self.coords = torch.stack(coords, dim=0)

        # pad regs 
        regs = [torch.nn.functional.pad(torch.from_numpy(reg) + 1, (0, new_size - reg.shape[0])) for reg in regs]
        self.regs = torch.stack(regs, dim=0)

        # save metadata to class
        metadata = sz_table.loc[indices]
        assert len(metadata) == len(signals)
        # reset index
        metadata.reset_index(inplace=True)
        self.metadata = metadata
        self.transform = transform

        self.tensor_transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=(5, 5))], p=0.6),
        ])

    def get_cropped_data(self, tensor, cropped_len):
        """
        Crop the input tensor to a specified length.

        Args:
            tensor (torch.Tensor): The input tensor to be cropped.
            cropped_len (int): The desired length of the cropped tensor.

        Returns:
            torch.Tensor: The cropped tensor.

        """
        choices = torch.randint(low=0, high=tensor.size()[-1] - cropped_len, size=(1,))
        start_indices = choices
        indices = (start_indices + torch.arange(cropped_len)).repeat(MODEL_CONFIG.channel_buffer_size, 1)
        cropped_tensor = tensor.gather(dim=1, index=indices)

        return cropped_tensor

    def multicrop(self, tensor):
        """
        Get multiple crops of the data in the time domain while preserving all the channels.

        Args:
            tensor (torch.Tensor): The input tensor containing the data.

        Returns:
            list: A list of tensors, each representing a cropped version of the input tensor.
        """
        size_list = MODEL_CONFIG.multi_crop_config
        if self.transform:
            data_list = [self.tensor_transform(self.get_cropped_data(tensor, sl).unsqueeze(0)).squeeze(0)
                         for sl in size_list]
        else:
            data_list = [self.get_cropped_data(tensor, sl) for sl in size_list]

        return data_list

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of signals in the dataset.
        """
        return len(self.signals)

    def __getitem__(self, idx):
        """
        Get the item at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the sample data, sample regs, and the index.
        """
        sample_data = self.signals[idx]
        sample_regs = self.regs[idx]
        sample_data = self.multicrop(sample_data)

        return sample_data, sample_regs, idx

# %%
class SzDatasetRegsFull(Dataset):
    def __init__(self, transform=False):
        """
        Initializes the Dataset object.

        Args:
            transform (bool): Flag indicating whether to apply data transformations. Default is False.
        """
        # load data
        fname = os.path.join(CONFIG.data_dir, "all_sz_coords_data_spatiotemporal_regs_10s.pkl")
        with open(fname, 'rb') as f:
            signals, coords, regs, ch_names, indices = pickle.load(f)

        # use torch padding to get the first dimension of signals and coords to be MODEL_CONFIG.channel_buffer_size
        new_size = MODEL_CONFIG.channel_buffer_size
        # pad signals
        signals = [torch.nn.functional.pad(torch.from_numpy(sig), (0, 0, 0, new_size - sig.shape[0])) for sig in signals]
        self.signals = torch.stack(signals, dim=0)

        # pad coords
        coords = [torch.nn.functional.pad(torch.from_numpy(coord), (0, 0, 0, new_size - coord.shape[0])) for coord in coords]
        self.coords = torch.stack(coords, dim=0)

        # pad regs 
        regs = [torch.nn.functional.pad(torch.from_numpy(reg) + 1, (0, new_size - reg.shape[0])) for reg in regs]
        self.regs = torch.stack(regs, dim=0)

        # save metadata to class
        metadata = sz_table.loc[indices]
        assert len(metadata) == len(signals)
        # reset index
        metadata.reset_index(inplace=True)
        self.metadata = metadata

        self.ch_names = ch_names
        

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of signals in the dataset.
        """
        return len(self.signals)

    def __getitem__(self, idx):
        """
        Get the item at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the sample data, sample regs, and the index.
        """
        sample_data = self.signals[idx]
        sample_regs = self.regs[idx]
        sample_ch_names = self.ch_names[idx]
        # sample_ch_names = np.array(sample_ch_names)
        # convert to list of strings
        sample_ch_names = [str(ch) for ch in sample_ch_names]

        return sample_data, sample_regs, sample_ch_names, idx

# %%
class SzDatasetRegsTenS(Dataset):
    def __init__(self, subsample=1, transform=False, random_seed=42, shuffle=False):
        """
        Initializes the Dataset object.

        Args:
            transform (bool): Flag indicating whether to apply data transformations. Default is False.
        """
        # load data
        fname = os.path.join(CONFIG.data_dir, "all_sz_coords_data_spatiotemporal_regs_10s.pkl")
        with open(fname, 'rb') as f:
            signals, coords, regs, ch_names, indices = pickle.load(f)

        # use torch padding to get the first dimension of signals and coords to be MODEL_CONFIG.channel_buffer_size
        new_size = int(MODEL_CONFIG.channel_buffer_size * subsample)

        if (subsample != 1) or shuffle:
            new_signals = []
            new_coords = []
            new_regs = []

            for sig, coord, reg in zip(signals, coords, regs):
                # subsample
                    # randomly select indices
                    selected_idx = torch.randint(low=0, high=sig.shape[0], size=(int(sig.shape[0] * subsample),), generator=torch.Generator().manual_seed(random_seed))
                    selected_idx, _ = torch.sort(selected_idx)
                    sig = sig[selected_idx]
                    coord = coord[selected_idx]
                    reg = reg[selected_idx]

                    new_signals.append(torch.nn.functional.pad(torch.from_numpy(sig), (0, 0, 0, new_size - sig.shape[0])))
                    new_coords.append(torch.nn.functional.pad(torch.from_numpy(coord), (0, 0, 0, new_size - coord.shape[0])))
                    new_regs.append(torch.nn.functional.pad(torch.from_numpy(reg) + 1, (0, new_size - reg.shape[0])))
            
            self.signals = torch.stack(new_signals, dim=0)
            self.coords = torch.stack(new_coords, dim=0)
            self.regs = torch.stack(new_regs, dim=0)

        else:
            # pad signals
            signals = [torch.nn.functional.pad(torch.from_numpy(sig), (0, 0, 0, new_size - sig.shape[0])) for sig in signals]
            self.signals = torch.stack(signals, dim=0)

            # pad coords
            coords = [torch.nn.functional.pad(torch.from_numpy(coord), (0, 0, 0, new_size - coord.shape[0])) for coord in coords]
            self.coords = torch.stack(coords, dim=0)

            # pad regs 
            regs = [torch.nn.functional.pad(torch.from_numpy(reg) + 1, (0, new_size - reg.shape[0])) for reg in regs]
            self.regs = torch.stack(regs, dim=0)

        # save metadata to class
        metadata = sz_table.loc[indices]
        assert len(metadata) == len(signals)
        # reset index
        metadata.reset_index(inplace=True)
        self.metadata = metadata
        self.transform = transform

        self.tensor_transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=(5, 5))], p=0.6),
        ])

        self.random_seed = random_seed
        self.subsample = subsample

    def get_cropped_data(self, tensor, cropped_len):
        """
        Crop the input tensor to a specified length.

        Args:
            tensor (torch.Tensor): The input tensor to be cropped.
            cropped_len (int): The desired length of the cropped tensor.

        Returns:
            torch.Tensor: The cropped tensor.

        """
        choices = torch.randint(low=0, high=tensor.size()[-1] - cropped_len, size=(1,))
        start_indices = choices
        indices = (start_indices + torch.arange(cropped_len)).repeat(MODEL_CONFIG.channel_buffer_size, 1)
        cropped_tensor = tensor.gather(dim=1, index=indices)

        return cropped_tensor

    def multicrop(self, tensor):
        """
        Get multiple crops of the data in the time domain while preserving all the channels.

        Args:
            tensor (torch.Tensor): The input tensor containing the data.

        Returns:
            list: A list of tensors, each representing a cropped version of the input tensor.
        """
        size_list = MODEL_CONFIG.multi_crop_config
        if self.transform:
            data_list = [self.tensor_transform(self.get_cropped_data(tensor, sl).unsqueeze(0)).squeeze(0)
                         for sl in size_list]
        else:
            data_list = [self.get_cropped_data(tensor, sl) for sl in size_list]

        return data_list

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of signals in the dataset.
        """
        return len(self.signals)

    def __getitem__(self, idx):
        """
        Get the item at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the sample data, sample regs, and the index.
        """
        sample_data = self.signals[idx]
        sample_regs = self.regs[idx]
        sample_data = self.multicrop(sample_data)

        # if self.subsample:
        #     n_zeros =

        return sample_data, sample_regs, idx



# %%
if __name__ == "__main__":
    dataset = SzDatasetRegs()

# %%
