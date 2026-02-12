"""
Test script: Extract ALL 10-second windows from ictal and interictal files for a single patient.

This script extracts every non-overlapping 10-second window from both ictal and interictal
EDF files, with explicit labels based on seizure timing:
- Ictal files: Pre-seizure (0-30s) = interictal (0), During seizure = ictal (1), Post-seizure = interictal (0)
- Interictal files: All windows = interictal (0)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

import numpy as np
import pandas as pd
import mne
from glob import glob
from tqdm import tqdm
import pickle
import h5py
from scipy.signal import resample_poly
from fractions import Fraction
from collections import defaultdict
import re

from config_benchmark import BenchmarkConfig


class AllWindowsExtractor:
    """Extract all possible 10-second windows from EDF files."""
    
    def __init__(self, config=BenchmarkConfig):
        """Initialize the extractor."""
        self.config = config
        self.target_sf = config.TARGET_SAMPLING_RATE
        self.window_duration = 10  # seconds
        
        # Load metadata
        self._load_metadata()
        
        # Load annotations for channel coordinates
        self._load_annotations()
    
    def _load_metadata(self):
        """Load seizure metadata and patient mappings."""
        # Load seizure times
        sz_times = pd.read_excel(
            self.config.SEIZURE_METADATA_FILE,
            sheet_name="AllSeizureTimes",
            index_col=0,
        )
        
        # Load RID-HUP mapping
        rid_hup_table = pd.read_csv(self.config.RID_HUP_TABLE_FILE, index_col=0)
        rid_hup_table.dropna(inplace=True, subset=["hupsubjno"])
        
        # Extract first 3 characters and convert to int
        rid_hup_table['hupsubjno'] = rid_hup_table['hupsubjno'].str[:3].astype(int)
        
        rid_hup_table.index = [f"sub-RID{x:04d}" for x in rid_hup_table.index]
        rid_hup_table['hupsubjno'] = [f"HUP{x:03d}" for x in rid_hup_table['hupsubjno']]
        
        # Create mappings
        self.rid_to_hup = rid_hup_table['hupsubjno'].to_dict()
        self.hup_to_rid = {v: k for k, v in self.rid_to_hup.items()}
        
        # Clean seizure times
        sz_times.dropna(inplace=True, subset=["IEEGname"])
        sz_times = sz_times[sz_times["IEEGname"] != "HUP203_phaseII"]
        
        # Create sz_table with RID indices
        sz_table = sz_times.copy()
        sz_table.index = sz_table.index.map(self.hup_to_rid)
        sz_table.reset_index(inplace=True)
        
        self.sz_table = sz_table
        
        print(f"✓ Loaded metadata for {len(sz_table)} seizures")
    
    def _load_annotations(self):
        """Load channel annotations (coordinates and regions)."""
        master_bipolars = pd.read_csv(self.config.ANNOTATIONS_FILE, index_col=0)
        
        # Load DKT mapping
        if os.path.exists(self.config.DKT_MAPPING_FILE):
            dkt_regs = pd.read_csv(self.config.DKT_MAPPING_FILE)
            dkt_to_custom = dict(zip(dkt_regs['atlas_index'], dkt_regs['atlas_index']))
        else:
            dkt_to_custom = {}
        
        if 'final_label' in master_bipolars.columns:
            master_bipolars['reg'] = master_bipolars.final_label.map(
                lambda x: dkt_to_custom.get(x, x) if pd.notna(x) else 0
            )
        else:
            master_bipolars['reg'] = 0
        
        self.annotations = master_bipolars
        
        print(f"✓ Loaded annotations for {len(master_bipolars)} channels")
    
    def _create_bipolar_channels(self, data, ch_names):
        """
        Convert monopolar channels to bipolar by taking differences between adjacent channels.
        
        Args:
            data: numpy array of shape (channels, samples)
            ch_names: list of channel names
        
        Returns:
            tuple: (bipolar_data, bipolar_ch_names)
        """
        electrode_groups = defaultdict(list)
        
        for idx, ch_name in enumerate(ch_names):
            # Extract electrode name (letters before numbers)
            match = re.match(r'([A-Za-z]+)(\d+)', ch_name)
            if match:
                electrode = match.group(1)
                number = int(match.group(2))
                electrode_groups[electrode].append((number, idx, ch_name))
        
        # Create bipolar pairs
        bipolar_data = []
        bipolar_names = []
        
        for electrode, channels in electrode_groups.items():
            # Sort by channel number
            channels.sort(key=lambda x: x[0])
            
            # Create consecutive pairs
            for i in range(len(channels) - 1):
                num1, idx1, name1 = channels[i]
                num2, idx2, name2 = channels[i + 1]
                
                # Only create pair if numbers are consecutive
                if num2 == num1 + 1:
                    # Bipolar signal is the difference
                    bipolar_signal = data[idx1] - data[idx2]
                    bipolar_name = f"{name1}-{name2}"
                    
                    bipolar_data.append(bipolar_signal)
                    bipolar_names.append(bipolar_name)
        
        if len(bipolar_data) == 0:
            return None, None
        
        bipolar_data = np.array(bipolar_data)
        return bipolar_data, bipolar_names
    
    def _match_coordinates(self, bipolar_names, patient_id):
        """
        Match bipolar channel names to coordinates and regions.
        
        Args:
            bipolar_names: list of bipolar channel names
            patient_id: patient identifier
        
        Returns:
            tuple: (coords, regs, valid_names) or (None, None, None)
        """
        try:
            patient_annots = self.annotations.loc[patient_id]
        except KeyError:
            return None, None, None
        
        coords = []
        regs = []
        valid_names = []
        
        for bipolar_name in bipolar_names:
            if bipolar_name in patient_annots['name'].values:
                ann = patient_annots[patient_annots['name'] == bipolar_name].iloc[0]
                if 'mni_x' in ann and 'mni_y' in ann and 'mni_z' in ann:
                    coords.append([ann['mni_x'], ann['mni_y'], ann['mni_z']])
                    regs.append(ann.get('reg', 0))
                    valid_names.append(bipolar_name)
        
        if len(coords) == 0:
            return None, None, None
        
        return np.array(coords), np.array(regs), valid_names
    
    def _extract_windows_from_ictal_file(self, edf_file, patient_id, seizure_start, seizure_end):
        """
        Extract all non-overlapping 10s windows from an ictal EDF file.
        
        Args:
            edf_file: Path to ictal EDF file
            patient_id: Patient identifier
            seizure_start: Seizure start time in original recording (seconds)
            seizure_end: Seizure end time in original recording (seconds)
        
        Returns:
            list of tuples: (signals, coords, regs, ch_names, label, metadata)
        """
        windows = []
        
        try:
            # Load EDF
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            total_duration = raw.times[-1]
            fs = raw.info['sfreq']
            ch_names = raw.ch_names
            data = raw.get_data()
            
            # Resample to target frequency if needed
            if fs != self.target_sf:
                frac = Fraction(int(self.target_sf), int(fs))
                data = resample_poly(data, frac.numerator, frac.denominator, axis=1)
            
            # Convert to bipolar
            bipolar_data, bipolar_names = self._create_bipolar_channels(data, ch_names)
            if bipolar_data is None:
                return windows
            
            # Match coordinates
            coords, regs, valid_names = self._match_coordinates(bipolar_names, patient_id)
            if coords is None:
                return windows
            
            # Filter to valid channels
            valid_indices = [i for i, name in enumerate(bipolar_names) if name in valid_names]
            bipolar_data = bipolar_data[valid_indices]
            
            # Calculate seizure timing in file
            # Files start 30 seconds before seizure, so seizure starts at 30s in file
            seizure_start_in_file = 30.0
            seizure_duration = seizure_end - seizure_start
            seizure_end_in_file = seizure_start_in_file + seizure_duration
            
            # Extract all non-overlapping 10s windows
            window_start = 0
            while window_start + self.window_duration <= total_duration:
                window_end = window_start + self.window_duration
                
                # Determine label based on timing
                if window_end <= seizure_start_in_file:
                    # Pre-ictal (interictal)
                    label = 0
                    label_desc = "pre-ictal"
                elif window_start >= seizure_end_in_file:
                    # Post-ictal (interictal)
                    label = 0
                    label_desc = "post-ictal"
                else:
                    # Ictal (during seizure)
                    label = 1
                    label_desc = "ictal"
                
                # Extract window samples
                start_sample = int(window_start * self.target_sf)
                end_sample = int(window_end * self.target_sf)
                window_data = bipolar_data[:, start_sample:end_sample]
                
                # Store window
                metadata = {
                    'file': os.path.basename(edf_file),
                    'window_start': window_start,
                    'window_end': window_end,
                    'seizure_start_in_file': seizure_start_in_file,
                    'seizure_end_in_file': seizure_end_in_file,
                    'label_desc': label_desc,
                    'file_type': 'ictal'
                }
                
                windows.append((
                    window_data.astype(np.float32),
                    coords.copy(),
                    regs.copy(),
                    valid_names.copy(),
                    label,
                    metadata
                ))
                
                window_start += self.window_duration
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(edf_file)}: {e}")
        
        return windows
    
    def _extract_windows_from_interictal_file(self, edf_file, patient_id):
        """
        Extract all non-overlapping 10s windows from an interictal EDF file.
        
        Args:
            edf_file: Path to interictal EDF file
            patient_id: Patient identifier
        
        Returns:
            list of tuples: (signals, coords, regs, ch_names, label, metadata)
        """
        windows = []
        
        try:
            # Load EDF
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            total_duration = raw.times[-1]
            fs = raw.info['sfreq']
            ch_names = raw.ch_names
            data = raw.get_data()
            
            # Resample to target frequency if needed
            if fs != self.target_sf:
                frac = Fraction(int(self.target_sf), int(fs))
                data = resample_poly(data, frac.numerator, frac.denominator, axis=1)
            
            # Convert to bipolar
            bipolar_data, bipolar_names = self._create_bipolar_channels(data, ch_names)
            if bipolar_data is None:
                return windows
            
            # Match coordinates
            coords, regs, valid_names = self._match_coordinates(bipolar_names, patient_id)
            if coords is None:
                return windows
            
            # Filter to valid channels
            valid_indices = [i for i, name in enumerate(bipolar_names) if name in valid_names]
            bipolar_data = bipolar_data[valid_indices]
            
            # Extract all non-overlapping 10s windows (all labeled as interictal)
            window_start = 0
            while window_start + self.window_duration <= total_duration:
                window_end = window_start + self.window_duration
                
                # Extract window samples
                start_sample = int(window_start * self.target_sf)
                end_sample = int(window_end * self.target_sf)
                window_data = bipolar_data[:, start_sample:end_sample]
                
                # Store window (all interictal)
                metadata = {
                    'file': os.path.basename(edf_file),
                    'window_start': window_start,
                    'window_end': window_end,
                    'label_desc': 'interictal',
                    'file_type': 'interictal'
                }
                
                windows.append((
                    window_data.astype(np.float32),
                    coords.copy(),
                    regs.copy(),
                    valid_names.copy(),
                    0,  # All interictal
                    metadata
                ))
                
                window_start += self.window_duration
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(edf_file)}: {e}")
        
        return windows
    
    def extract_patient(self, patient_id):
        """
        Extract all windows for a single patient from both ictal and interictal files.
        
        Args:
            patient_id: Patient identifier (e.g., 'sub-RID0106')
        
        Returns:
            dict with lists of signals, coords, regs, ch_names, labels, patient_ids, window_info
        """
        all_windows = []
        
        print(f"\n{'='*60}")
        print(f"Extracting windows for patient: {patient_id}")
        print(f"{'='*60}")
        
        # Process ictal files
        patient_seizures = self.sz_table[self.sz_table['Patient'] == patient_id]
        print(f"\nProcessing {len(patient_seizures)} ictal files...")
        
        ictal_stats = {'pre_ictal': 0, 'ictal': 0, 'post_ictal': 0}
        
        for idx, seizure in patient_seizures.iterrows():
            # Find ictal EDF file
            edf_files = glob(
                os.path.join(
                    self.config.RAW_DATA_DIR,
                    patient_id,
                    "ses-clinical01/ieeg",
                    f"{patient_id}_ses-clinical01_task-ictal{int(seizure.start)}_*.edf"
                )
            )
            
            if len(edf_files) != 1:
                continue
            
            edf_file = edf_files[0]
            windows = self._extract_windows_from_ictal_file(
                edf_file, patient_id, seizure.start, seizure.end
            )
            
            # Count by type
            file_stats = {'pre_ictal': 0, 'ictal': 0, 'post_ictal': 0}
            for win in windows:
                label_desc = win[5]['label_desc']
                if label_desc == 'pre-ictal':
                    file_stats['pre_ictal'] += 1
                    ictal_stats['pre_ictal'] += 1
                elif label_desc == 'ictal':
                    file_stats['ictal'] += 1
                    ictal_stats['ictal'] += 1
                elif label_desc == 'post-ictal':
                    file_stats['post_ictal'] += 1
                    ictal_stats['post_ictal'] += 1
            
            if len(windows) > 0:
                print(f"  {os.path.basename(edf_file)}: "
                      f"{file_stats['pre_ictal']} pre-ictal, "
                      f"{file_stats['ictal']} ictal, "
                      f"{file_stats['post_ictal']} post-ictal")
            
            all_windows.extend(windows)
        
        print(f"\nIctal files summary:")
        print(f"  Pre-ictal windows: {ictal_stats['pre_ictal']}")
        print(f"  Ictal windows: {ictal_stats['ictal']}")
        print(f"  Post-ictal windows: {ictal_stats['post_ictal']}")
        print(f"  Total from ictal files: {sum(ictal_stats.values())}")
        
        # Process interictal files
        patient_dir = os.path.join(
            self.config.RAW_DATA_DIR,
            patient_id,
            "ses-clinical01/ieeg"
        )
        
        interictal_files = glob(
            os.path.join(patient_dir, f"{patient_id}_ses-clinical01_task-interictal*_ieeg.edf")
        )
        
        print(f"\nProcessing {len(interictal_files)} interictal files...")
        
        interictal_count = 0
        for edf_file in tqdm(interictal_files, desc="  Interictal files"):
            windows = self._extract_windows_from_interictal_file(edf_file, patient_id)
            interictal_count += len(windows)
            all_windows.extend(windows)
        
        print(f"  Total from interictal files: {interictal_count}")
        
        # Compile results
        print(f"\n{'='*60}")
        print(f"Extraction complete for {patient_id}")
        print(f"{'='*60}")
        print(f"  Total windows: {len(all_windows)}")
        print(f"  Interictal (label=0): {sum(1 for w in all_windows if w[4] == 0)}")
        print(f"  Ictal (label=1): {sum(1 for w in all_windows if w[4] == 1)}")
        
        if len(all_windows) > 0:
            print(f"\n  Sample window shape: {all_windows[0][0].shape}")
            print(f"  Sample coords shape: {all_windows[0][1].shape}")
            print(f"  Sample regs shape: {all_windows[0][2].shape}")
            print(f"  Sample channel count: {len(all_windows[0][3])}")
        
        # Convert to output format
        output = {
            'signals': [w[0] for w in all_windows],
            'coords': [w[1] for w in all_windows],
            'regs': [w[2] for w in all_windows],
            'ch_names': [w[3] for w in all_windows],
            'labels': [w[4] for w in all_windows],
            'patient_ids': [patient_id] * len(all_windows),
            'window_info': [w[5] for w in all_windows]
        }
        
        return output


def main():
    """Test extraction on a single patient."""
    print("="*60)
    print("Test: Extract All Windows - Single Patient")
    print("="*60)
    
    # Create config
    config = BenchmarkConfig()
    
    # Create extractor
    extractor = AllWindowsExtractor(config)
    
    # Get first patient
    unique_patients = extractor.sz_table['Patient'].unique()
    test_patient = unique_patients[0]
    
    # Extract all windows
    output = extractor.extract_patient(test_patient)
    
    # Save for inspection in test directory
    test_dir = os.path.join(
        config.OUTPUT_DATA_DIR,
        "test_all_windows"
    )
    os.makedirs(test_dir, exist_ok=True)
    
    output_file = os.path.join(test_dir, f"{test_patient}.h5")
    
    # Use simple HDF5 save for test (matches per-patient format)
    n_samples = len(output['signals'])
    n_channels = output['signals'][0].shape[0]
    n_timesteps = output['signals'][0].shape[1]
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('signals', data=np.array(output['signals']), compression='gzip')
        f.create_dataset('coords', data=np.array(output['coords']), compression='gzip')
        f.create_dataset('regs', data=np.array(output['regs']), compression='gzip')
        f.create_dataset('labels', data=np.array(output['labels']))
        
        import json
        # Save patient ID
        f.create_dataset('patient_id', data=test_patient.encode('utf-8'), dtype=h5py.string_dtype())
        
        ch_names_json = [json.dumps(names).encode('utf-8') for names in output['ch_names']]
        f.create_dataset('ch_names', data=ch_names_json, dtype=h5py.string_dtype())
        
        window_info_json = [json.dumps(info).encode('utf-8') for info in output['window_info']]
        f.create_dataset('window_info', data=window_info_json, dtype=h5py.string_dtype())
    
    print(f"\n✓ Saved test data to: {output_file}")
    print(f"  Directory structure matches full extraction (one file per patient)")


if __name__ == "__main__":
    main()
