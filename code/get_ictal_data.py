# %%
# pylint: disable=C0103
"""
    This script will get the ictal clips from the iEEG data and save them in BIDS format in the common data folder.
    Run on February 27, 2023 by Akash Pattnaik on pioneer server.   

    Rerun on March 21, 2023 by Akash Pattnaik on pioneer server to include 30 seconds preictal
"""
# imports
import warnings
import logging

import numpy as np
import pandas as pd
from utils import get_iEEG_data, clean_labels, check_channel_types
from mne_bids import BIDSPath, write_raw_bids
from tqdm import tqdm
import mne

from config import CONFIG

OVERWRITE = True
# %%
# logging
logging.basicConfig(
    filename="/mnt/leif/littlab/users/pattnaik/ictal_patterns/code/logs/script03-ictal_clips.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)
logger = logging.getLogger()

# %%
# paths
bids_path_kwargs = {
    "root": CONFIG.bids_dir,
    "datatype": "ieeg",
    "extension": ".edf",
    "suffix": "ieeg",
    "task": "ictal",
    "session": "clinical01",
}
bids_path = BIDSPath(**bids_path_kwargs)

# %%
# get ieeg data function kwargs
ieeg_kwargs = {
    "username": CONFIG.usr,
    "password_bin_file": CONFIG.pwd,
}
# %%
# iterate through table, get the data, and save the clips in bids format
for ieeg_fname, group in tqdm(
    CONFIG.sz_times.groupby("IEEGname"),
    total=CONFIG.sz_times["IEEGname"].nunique(),
    desc="subjects",
    position=0,
):
    # sort by start time
    group = group.sort_values("start")
    group.reset_index(inplace=True, drop=True)
    for idx, row in tqdm(
        group.iterrows(), total=group.shape[0], desc="seizures", position=1, leave=False
    ):
        # get RID
        hupno = int(row["IEEGname"].split("_")[0][3:])
        rid = CONFIG.rid_hup_table[CONFIG.rid_hup_table["hupsubjno"] == hupno].index[0]

        # swap start and end if start is after end
        if row["start"] > row["end"]:
            row["start"], row["end"] = row["end"], row["start"]

        # get bids path
        sz_clip_bids_path = bids_path.copy().update(
            subject=f"RID{rid:04d}",
            run=int(row["IEEGID"]),
            task=f"ictal{int(row['start'])}",
        )

        # check if the file already exists, if so, skip
        if sz_clip_bids_path.fpath.exists() and not OVERWRITE:
            continue

        # HUP097 does not have an end time, so we'll just use 60 seconds from the start
        if np.isnan(row["end"]):
            row["end"] = row["start"] + 60

        # get the duration and clip it to 5 mins
        round_duration = np.ceil(row["end"] - row["start"])
        round_duration = np.min([round_duration, 270])

        data, fs = get_iEEG_data(
            iEEG_filename=row["IEEGname"],
            start_time_usec=(row["start"] - 30) * 1e6, # start 30 seconds before the seizure
            stop_time_usec=(row["start"] + round_duration) * 1e6,
            **ieeg_kwargs,
        )

        # channels with flat line may not save proprely, so we'll drop them
        data = data[data.columns[data.min(axis=0) != data.max(axis=0)]]

        # clean the labels
        data.columns = clean_labels(data.columns, pt=ieeg_fname)

        # if there are duplicate labels, keep the first one in the table
        data = data.loc[:, ~data.columns.duplicated()]
        # get the channel types
        ch_types = check_channel_types(list(data.columns))
        ch_types.set_index("name", inplace=True, drop=True)

        # convert nan to 0
        data.fillna(0, inplace=True)

        # save the data
        # run is the iEEG file number
        # task is ictal with the start time in seconds appended
        data_info = mne.create_info(
            ch_names=list(data.columns), sfreq=fs, ch_types="eeg", verbose=False
        )
        raw = mne.io.RawArray(
            data.to_numpy().T / 1e6,  # mne needs data in volts,
            data_info,
            verbose=False,
        )
        raw.set_channel_types(ch_types.type)
        annots = mne.Annotations(
            onset=[30], # seizure starts 30 seconds after the start of the clip
            duration=[round_duration],
            description=["ictal"],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_annotations(annots)

            write_raw_bids(
                raw,
                sz_clip_bids_path,
                overwrite=OVERWRITE,
                verbose=False,
                allow_preload=True,
                format="EDF",
            )
# %%
