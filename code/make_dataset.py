# %%
from os.path import join as ospj
import pandas as pd
import numpy as np
import mne
from glob import glob
import os
from tqdm import tqdm
from fractions import Fraction
from scipy.signal import resample_poly
import pickle

from config import CONFIG
from utils import *

TARGET_SF = 256
# CLIP_SIZE = 45  # seconds
CLIP_SIZE = 10  # seconds 
RAW_DATA_DIR = "/mnt/leif/littlab/data/Human_Data/CNT_iEEG_BIDS"

# make sz_table
sz_table = CONFIG.sz_times.copy()
sz_table.index = sz_table.index.map(CONFIG.hup_to_rid)
# move index to column and reindex
sz_table.reset_index(inplace=True)

master_bipolars = pd.read_csv(CONFIG.annotations_file, index_col=0)
master_bipolars['reg'] = master_bipolars.final_label.map(CONFIG.dkt_to_custom)

regs_li = pd.read_csv(ospj(CONFIG.data_dir, 'atlases', 'luts', 'dktg_reordered.csv'))
regs_li = regs_li.atlas_index.values
regs = pd.DataFrame(regs_li, columns=['reg'])
# %%
all_data_fname = os.path.join(
                CONFIG.data_dir,
                "all_sz_coords_data_spatiotemporal_regs_10s.pkl"
            )

if True:
    all_sz_data = []
    all_coords_data = []
    all_regs_data = []
    present_indices = []
    missing_idx = []
    all_ch_names = []

    print("Loading data...")
    for ind, row in tqdm(sz_table.iterrows(), total=len(sz_table)):
        fname = glob(
            ospj(
                RAW_DATA_DIR,
                row.Patient,
                "ses-clinical01/ieeg",
                f"{row.Patient}_ses-clinical01_task-ictal{int(row.start)}_*.edf")
        )
        if len(fname) == 0:
            missing_idx.append([ind, "No file found"])
            continue
        if len(fname) > 1:
            missing_idx.append([ind, "Multiple files found"])
            continue
        else:
            fname = fname[0]

        try:
            data = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        except:
            continue
        signals = data.get_data()  # (n_channels, n_samples)

        fs = data.info["sfreq"]
        ch_names = data.ch_names
        ch_names = clean_labels(ch_names, pt=None)
        ch_types = check_channel_types(ch_names)

        # notch filter at 60 Hz
        signals = notch_filter(signals.T, fs).T  # (n_channels, n_samples)

        # low pass filter with 10th order butterworth filter
        signals = bandpass_filter(
            signals, fs, order=10, lo=0.5, hi=120
        )  # (n_channels, n_samples)

        # apply bipolar montage
        signals, bipolar_ch_names = bipolar_montage(
            signals, ch_types
        )  # (n_channels, n_samples)
        signals = pd.DataFrame(
            signals, index=bipolar_ch_names.name
        ).T  # (n_samples, n_channels)

        # downsample to 200 Hz
        frac = Fraction(TARGET_SF, int(fs))
        signals = resample_poly(
            signals, up=frac.numerator, down=frac.denominator
        )  # (n_samples, n_channels)

        # format signals into edf
        signals = np.array(signals).T  # (n_channels, n_samples)
        ch_types_for_edf = ["eeg"] * len(bipolar_ch_names)
        new_info = mne.create_info(
            bipolar_ch_names.name.to_list(), TARGET_SF, ch_types_for_edf,
            verbose=False
        )
        new_data = mne.io.RawArray(signals, new_info, verbose=False)

        # if there's an eeg annotation onset that starts at 30 seconds, then use 
        # 15 seconds before to 30 seconds after as the seizure clip
        # otherwise, use 45 seconds from the start of the clip
        if data.annotations.onset[0] == 30:
            start = 30
        else:
            start = 0

        # if the new_data is less than 60 seconds, then skip it
        if new_data.n_times < 60 * TARGET_SF:
            missing_idx.append([ind, "Less than 60 seconds"])
            continue

        # get the clip
        clip = new_data.copy().crop(
            tmin=start, tmax=start + CLIP_SIZE, include_tmax=False, verbose=False
        )

        # get the corresponding rows of the master bipolars
        pt_bipolars = master_bipolars.loc[
            master_bipolars.index == row.Patient
            ].copy()

        if len(pt_bipolars) == 0:
            print("No elecs in master bipolars")
            missing_idx.append([ind, "No elecs in master bipolars"])
            continue

        # keep the rows of pt_bipolars that have name in clip.ch_names
        pt_bipolars = pt_bipolars.loc[
            pt_bipolars.name.isin(clip.ch_names)
        ].copy()
        pt_bipolars = pt_bipolars.loc[
            pt_bipolars.reg.isin(regs.reg)
        ]

        # scale the clip to zero mean and unit variance
        # clip = clip.apply_function(lambda x: (x - x.mean()) / x.std())
        clip = clip.apply_function(lambda x: (x - np.nanmean(x)) / np.nanstd(x))

        # make sure the clip has the same channels as pt_bipolars
        clip = clip.pick_channels(pt_bipolars.name.to_list())

        # convert clip data to float16
        clip_data = clip.get_data().astype(np.float16)
        clip_coords = pt_bipolars[["mni_x", "mni_y", "mni_z"]].values
        clip_reg = pt_bipolars["reg"].values

        ch_names = clip.ch_names

        clip_data = clip_data[np.isin(clip_reg, regs.reg.values)]
        clip_coords = clip_coords[np.isin(clip_reg, regs.reg.values)]
        clip_reg = clip_reg[np.isin(clip_reg, regs.reg.values)]
        clip_reg_idx = np.array([np.where(regs.reg.values == i)[0][0] for i in clip_reg])
        ch_names = np.array(ch_names)[np.isin(clip_reg, regs.reg.values)]

        if len(clip_reg_idx) == 0:
            print("No reg in regs")
            missing_idx.append([ind, "No reg in regs"])
            continue

        all_ch_names.append(ch_names)
        all_sz_data.append(clip_data)
        all_coords_data.append(clip_coords)
        all_regs_data.append(clip_reg_idx)
        present_indices.append(ind)
    print("Saving data...")

    # # pickle the data
    with open(all_data_fname, 'wb') as f:
        pickle.dump([all_sz_data, all_coords_data, all_regs_data, all_ch_names, present_indices], f)
    regs.to_csv(os.path.join(CONFIG.data_dir, "unique_regs.csv"), index=False)

# %%
