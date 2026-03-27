# %%
from os.path import join as ospj
import pandas as pd
import numpy as np
import mne
from glob import glob
import os
from math import ceil
from tqdm import tqdm
from fractions import Fraction
from scipy.signal import resample_poly
import pickle

from config import CONFIG
from utils import *

TARGET_SF = 256
RAW_DATA_DIR = r"e:\projects\ieeg_sz_embedding\CNT_iEEG_BIDS_local"

# make sz_table
sz_table = CONFIG.sz_times.copy()
sz_table.index = sz_table.index.map(CONFIG.hup_to_rid)
sz_table.reset_index(inplace=True)

master_bipolars = pd.read_csv(CONFIG.annotations_file, index_col=0)
master_bipolars['reg'] = master_bipolars.final_label.map(CONFIG.dkt_to_custom)

regs_li = pd.read_csv(ospj(CONFIG.data_dir, 'atlases', 'luts', 'dktg_reordered.csv'))
regs_li = regs_li.atlas_index.values
regs = pd.DataFrame(regs_li, columns=['reg'])


# ---------------------------------------------------------------------------
# Clip-bounds helper
# ---------------------------------------------------------------------------

def get_clip_bounds(mode, row, onset_sec):
    """
    Return (tmin, tmax, min_file_duration) for the requested clip mode.

    Parameters
    ----------
    mode : str
        One of '45s', '20s', '20s_peri', 'full'.
    row : pandas Series
        Row from sz_table; must have 'start' and 'end' columns.
    onset_sec : float
        Time (seconds) of seizure onset within the EDF file (typically 30.0).

    Returns
    -------
    tuple (tmin, tmax, min_duration) or None if the clip cannot be formed.
    """
    if mode == "45s":
        tmin = onset_sec
        tmax = onset_sec + 45
        min_dur = tmax
    elif mode == "20s":
        tmin = onset_sec
        tmax = onset_sec + 20
        min_dur = tmax
    elif mode == "20s_peri":
        tmin = onset_sec - 10
        tmax = onset_sec + 10
        if tmin < 0:
            return None  # no pre-ictal buffer available
        min_dur = tmax
    elif mode == "full":
        end = row.get("end", np.nan) if hasattr(row, "get") else getattr(row, "end", np.nan)
        if pd.isna(end):
            end = row.start + 60
        sz_duration = min(ceil(end - row.start), 270)
        tmin = onset_sec
        tmax = onset_sec + sz_duration
        min_dur = tmax
    else:
        raise ValueError(f"Unknown clip mode: {mode}")

    return tmin, tmax, min_dur


# ---------------------------------------------------------------------------
# Per-seizure preprocessing
# ---------------------------------------------------------------------------

def preprocess_seizure(fname):
    """
    Load an EDF, apply notch + bandpass + bipolar montage + resample.

    Returns
    -------
    (new_data : mne.io.RawArray, onset_sec : float) or None on failure.
    onset_sec is the seizure onset time within the file (seconds).
    """
    try:
        data = mne.io.read_raw_edf(fname, preload=True, verbose=False)
    except Exception:
        return None

    signals = data.get_data()  # (n_channels, n_samples)
    fs = data.info["sfreq"]
    ch_names = data.ch_names
    ch_names = clean_labels(ch_names, pt=None)
    ch_types = check_channel_types(ch_names)

    signals = notch_filter(signals.T, fs).T
    signals = bandpass_filter(signals, fs, order=10, lo=0.5, hi=120)

    signals, bipolar_ch_names = bipolar_montage(signals, ch_types)
    signals = pd.DataFrame(signals, index=bipolar_ch_names.name).T

    frac = Fraction(TARGET_SF, int(fs))
    signals = resample_poly(signals, up=frac.numerator, down=frac.denominator)

    signals = np.array(signals).T  # (n_channels, n_samples)
    ch_types_for_edf = ["eeg"] * len(bipolar_ch_names)
    new_info = mne.create_info(
        bipolar_ch_names.name.to_list(), TARGET_SF, ch_types_for_edf, verbose=False
    )
    new_data = mne.io.RawArray(signals, new_info, verbose=False)

    onset_sec = 30.0 if data.annotations.onset[0] == 30 else 0.0

    return new_data, onset_sec


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(mode, output_fname):
    """
    Build and save a dataset pkl for the given clip mode.

    The pkl contains:
        [all_sz_data, all_coords_data, all_regs_data, all_ch_names, present_indices]
    """
    print(f"\n{'='*60}")
    print(f"Running pipeline: mode={mode!r}  ->  {output_fname}")
    print(f"{'='*60}")

    all_sz_data = []
    all_coords_data = []
    all_regs_data = []
    present_indices = []
    missing_idx = []
    all_ch_names = []

    for ind, row in tqdm(sz_table.iterrows(), total=len(sz_table)):
        fname_list = glob(
            ospj(
                RAW_DATA_DIR,
                row.Patient,
                "ses-clinical01/ieeg",
                f"{row.Patient}_ses-clinical01_task-ictal{int(row.start)}_*.edf"
            )
        )
        if len(fname_list) == 0:
            missing_idx.append([ind, "No file found"])
            continue
        if len(fname_list) > 1:
            missing_idx.append([ind, "Multiple files found"])
            continue
        fname = fname_list[0]

        result = preprocess_seizure(fname)
        if result is None:
            missing_idx.append([ind, "Preprocessing failed"])
            continue
        new_data, onset_sec = result

        # Compute crop window for this mode
        bounds = get_clip_bounds(mode, row, onset_sec)
        if bounds is None:
            missing_idx.append([ind, f"Cannot form clip for mode {mode} (no pre-ictal buffer)"])
            continue
        tmin, tmax, min_dur = bounds

        # Check the file is long enough
        if new_data.n_times < min_dur * TARGET_SF:
            missing_idx.append([ind, f"File shorter than required {min_dur}s"])
            continue

        # Get matching electrode metadata
        pt_bipolars = master_bipolars.loc[
            master_bipolars.index == row.Patient
        ].copy()

        if len(pt_bipolars) == 0:
            missing_idx.append([ind, "No elecs in master bipolars"])
            continue

        # Crop to the desired window
        clip = new_data.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False, verbose=False)

        pt_bipolars = pt_bipolars.loc[pt_bipolars.name.isin(clip.ch_names)].copy()
        pt_bipolars = pt_bipolars.loc[pt_bipolars.reg.isin(regs.reg)]

        clip = clip.apply_function(lambda x: (x - np.nanmean(x)) / np.nanstd(x))
        clip = clip.pick_channels(pt_bipolars.name.to_list())

        clip_data = clip.get_data().astype(np.float16)
        clip_coords = pt_bipolars[["mni_x", "mni_y", "mni_z"]].values
        clip_reg = pt_bipolars["reg"].values
        ch_names_out = np.array(clip.ch_names)

        mask = np.isin(clip_reg, regs.reg.values)
        clip_data = clip_data[mask]
        clip_coords = clip_coords[mask]
        clip_reg = clip_reg[mask]
        ch_names_out = ch_names_out[mask]
        clip_reg_idx = np.array([np.where(regs.reg.values == i)[0][0] for i in clip_reg])

        if len(clip_reg_idx) == 0:
            missing_idx.append([ind, "No reg in regs"])
            continue

        all_ch_names.append(ch_names_out)
        all_sz_data.append(clip_data)
        all_coords_data.append(clip_coords)
        all_regs_data.append(clip_reg_idx)
        present_indices.append(ind)

    print(f"Saving {len(all_sz_data)} seizures to {output_fname} ...")
    with open(output_fname, 'wb') as f:
        pickle.dump(
            [all_sz_data, all_coords_data, all_regs_data, all_ch_names, present_indices], f
        )
    print(f"Done. Missing/skipped: {len(missing_idx)}")
    return missing_idx


# ---------------------------------------------------------------------------
# Run all clip modes
# ---------------------------------------------------------------------------

CLIP_MODES = {
    "45s":      "all_sz_coords_data_spatiotemporal_regs_45s.pkl",
    "20s":      "all_sz_coords_data_spatiotemporal_regs_20s.pkl",
    "20s_peri": "all_sz_coords_data_spatiotemporal_regs_20s_peri.pkl",
    "full":     "all_sz_coords_data_spatiotemporal_regs_full.pkl",
}

for mode, fname in CLIP_MODES.items():
    run_pipeline(mode, os.path.join(CONFIG.data_dir, fname))

regs.to_csv(os.path.join(CONFIG.data_dir, "unique_regs.csv"), index=False)

# %%
