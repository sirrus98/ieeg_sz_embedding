"""
Extract ALL 10-second windows from ictal and interictal files for all patients.

This script follows the EXACT preprocessing pipeline from code/make_dataset.py:
1. Load EDF with mne.io.read_raw_edf()
2. Clean channel labels using clean_labels()
3. Get channel types using check_channel_types()
4. Apply 60Hz notch filter using notch_filter()
5. Apply bandpass filter (0.5-120Hz, order=10) using bandpass_filter()
6. Apply bipolar montage using bipolar_montage()
7. Resample to 256Hz using resample_poly()
8. Per-channel z-score normalization: (x - nanmean(x)) / nanstd(x)
9. Match channels to master_bipolars annotations
10. Filter to valid regions from unique_regs.csv
11. Convert to float16

Window Extraction:
- Ictal files: Pre-ictal (0-30s) = interictal (0), During seizure = ictal (1), Post-ictal = interictal (0)
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
import json
from scipy.signal import resample_poly
from fractions import Fraction

from config_benchmark import BenchmarkConfig
from utils import check_channel_types, clean_labels, notch_filter, bandpass_filter, bipolar_montage


class AllWindowsExtractor:
    """Extract all possible 10-second windows from EDF files using make_dataset.py preprocessing."""
    
    def __init__(self, config=BenchmarkConfig, use_global_norm=False, global_stats_file=None):
        """
        Initialize the extractor.
        
        Args:
            config: BenchmarkConfig instance
            use_global_norm: If True, use global mean/std for normalization. If False, use per-channel.
            global_stats_file: Path to precomputed global statistics file (JSON)
        """
        self.config = config
        self.target_sf = config.TARGET_SAMPLING_RATE
        self.window_duration = 10  # seconds
        self.use_global_norm = use_global_norm
        
        # Load metadata
        self._load_metadata()
        
        # Load annotations and region mappings
        self._load_annotations()
        
        # Load or compute global statistics if needed
        if use_global_norm:
            if global_stats_file and os.path.exists(global_stats_file):
                self.global_stats = self._load_global_stats(global_stats_file)
                print(f"✓ Loaded global statistics from {global_stats_file}")
            else:
                print(f"✗ Global normalization requested but stats file not found: {global_stats_file}")
                print(f"  Please run: python extract_all_windows.py --compute-global-stats")
                raise FileNotFoundError(f"Global stats file not found: {global_stats_file}")
        else:
            self.global_stats = None
        
        print(f"✓ Initialization complete")
        print(f"  Normalization mode: {'Global' if use_global_norm else 'Per-channel'}")
    
    def _load_metadata(self):
        """Load seizure metadata and patient mappings (from make_dataset.py lines 22-26)."""
        # Load seizure times
        sz_times = pd.read_excel(
            self.config.SEIZURE_METADATA_FILE,
            sheet_name="AllSeizureTimes",
            index_col=0,
        )
        
        # Load RID-HUP mapping
        rid_hup_table = pd.read_csv(self.config.RID_HUP_TABLE_FILE, index_col=0)
        rid_hup_table.dropna(inplace=True, subset=["hupsubjno"])
        rid_hup_table['hupsubjno'] = rid_hup_table['hupsubjno'].str[:3].astype(int)
        rid_hup_table.index = [f"sub-RID{x:04d}" for x in rid_hup_table.index]
        rid_hup_table['hupsubjno'] = [f"HUP{x:03d}" for x in rid_hup_table['hupsubjno']]
        
        # Create mappings
        self.rid_to_hup = rid_hup_table['hupsubjno'].to_dict()
        self.hup_to_rid = {v: k for k, v in self.rid_to_hup.items()}
        
        # Clean seizure times
        sz_times.dropna(inplace=True, subset=["IEEGname"])
        sz_times = sz_times[sz_times["IEEGname"] != "HUP203_phaseII"]
        
        # Create sz_table with RID indices (from make_dataset.py lines 22-25)
        sz_table = sz_times.copy()
        sz_table.index = sz_table.index.map(self.hup_to_rid)
        sz_table.reset_index(inplace=True)
        
        self.sz_table = sz_table
        
        print(f"✓ Loaded metadata for {len(sz_table)} seizures")
    
    def _load_annotations(self):
        """Load channel annotations and region mappings (from make_dataset.py lines 27-32)."""
        # Load master bipolars (line 27-28)
        master_bipolars = pd.read_csv(self.config.ANNOTATIONS_FILE, index_col=0)
        
        # Load DKT to custom region mapping (from make_dataset.py lines 28-30)
        # This is identical to CONFIG.dkt_to_custom but loaded directly to avoid import issues
        dkt_mni_file = os.path.join(self.config.DATA_DIR, 'metadata', 'dkt_mni_parcs_RG.xlsx')
        if not os.path.exists(dkt_mni_file):
            raise FileNotFoundError(f"DKT mapping file not found: {dkt_mni_file}")
        
        dkt_mni_parcs = pd.read_excel(dkt_mni_file, header=None, sheet_name='Sheet1')
        dkt_custom = dkt_mni_parcs.iloc[:, [0, 2]]
        dkt_custom.columns = ['dkt', 'custom']
        dkt_custom = dkt_custom[dkt_custom['dkt'].str.startswith('Label')]
        dkt_custom['dkt_id'] = dkt_custom['dkt'].str.split(':').str[0].str.strip()
        dkt_custom['dkt_id'] = dkt_custom['dkt_id'].str.split(' ').str[1].str.strip().astype(int)
        dkt_custom['dkt'] = dkt_custom['dkt'].str.split(':').str[1].str.strip()
        dkt_custom.dropna(inplace=True)
        dkt_to_custom = dict(zip(dkt_custom['dkt'], dkt_custom['custom']))
        
        # Map regions
        master_bipolars['reg'] = master_bipolars.final_label.map(dkt_to_custom)
        
        # Load valid regions (lines 30-32)
        regs_file = os.path.join(self.config.DATA_DIR, 'unique_regs.csv')
        if not os.path.exists(regs_file):
            raise FileNotFoundError(f"Region file not found: {regs_file}")
        
        regs = pd.read_csv(regs_file)
        
        self.master_bipolars = master_bipolars
        self.regs = regs
        
        print(f"✓ Loaded annotations for {len(master_bipolars)} channels")
        print(f"✓ Loaded {len(self.regs)} valid regions")
    
    def _load_global_stats(self, stats_file):
        """Load precomputed global statistics from JSON file."""
        import json
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        return {
            'mean': stats['global_mean'],
            'std': stats['global_std'],
            'n_samples': stats['n_samples_used'],
            'n_files': stats['n_files_sampled']
        }
    
    def _save_global_stats(self, stats, stats_file):
        """Save global statistics to JSON file."""
        import json
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved global statistics to {stats_file}")
    
    def _preprocess_edf(self, edf_file, patient_id):
        """
        Apply full preprocessing pipeline from make_dataset.py (lines 66-105).
        
        Returns:
            tuple: (processed_raw_mne, bipolar_ch_names_df) or (None, None) if failed
        """
        try:
            # Step 1: Load EDF (line 66)
            data = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            signals = data.get_data()  # (n_channels, n_samples)
            
            fs = data.info["sfreq"]
            ch_names = data.ch_names
            
            # Step 2: Clean channel labels (line 73)
            ch_names = clean_labels(ch_names, pt=None)
            
            # Step 3: Get channel types (line 74)
            ch_types = check_channel_types(ch_names)
            
            # Step 4: Apply 60Hz notch filter (line 77)
            signals = notch_filter(signals.T, fs).T  # (n_channels, n_samples)
            
            # Step 5: Apply bandpass filter (lines 80-82)
            signals = bandpass_filter(
                signals, fs, order=10, lo=0.5, hi=120
            )  # (n_channels, n_samples)
            
            # Step 6: Apply bipolar montage (lines 85-87)
            signals, bipolar_ch_names = bipolar_montage(
                signals, ch_types
            )  # (n_channels, n_samples)
            
            # Convert to DataFrame for resampling (lines 88-90)
            signals = pd.DataFrame(
                signals, index=bipolar_ch_names.name
            ).T  # (n_samples, n_channels)
            
            # Step 7: Resample to 256Hz (lines 93-96)
            frac = Fraction(self.target_sf, int(fs))
            signals = resample_poly(
                signals, up=frac.numerator, down=frac.denominator
            )  # (n_samples, n_channels)
            
            # Step 8: Format signals into MNE RawArray (lines 99-105)
            signals = np.array(signals).T  # (n_channels, n_samples)
            ch_types_for_edf = ["eeg"] * len(bipolar_ch_names)
            new_info = mne.create_info(
                bipolar_ch_names.name.to_list(), self.target_sf, ch_types_for_edf,
                verbose=False
            )
            new_data = mne.io.RawArray(signals, new_info, verbose=False)
            
            return new_data, bipolar_ch_names
            
        except Exception as e:
            print(f"    ERROR preprocessing {os.path.basename(edf_file)}: {str(e)[:100]}")
            return None, None
    
    def compute_global_stats(self, n_files=200, output_file=None):
        """
        Compute global mean and std from randomly sampled EDF files.
        
        Args:
            n_files: Number of EDF files to sample (default: 200)
            output_file: Path to save statistics JSON file
            
        Returns:
            dict: Global statistics {'global_mean', 'global_std', 'n_samples_used', 'n_files_sampled'}
        """
        print(f"\n{'='*60}")
        print(f"Computing Global Statistics from {n_files} Random EDF Files")
        print(f"{'='*60}\n")
        
        # Collect all EDF files from all patients
        all_edf_files = []
        unique_patients = self.sz_table['Patient'].unique()
        
        for patient_id in unique_patients:
            # Ictal files
            patient_seizures = self.sz_table[self.sz_table['Patient'] == patient_id]
            for idx, seizure in patient_seizures.iterrows():
                edf_files = glob(
                    os.path.join(
                        self.config.RAW_DATA_DIR,
                        patient_id,
                        "ses-clinical01",
                        "ieeg",
                        f"{patient_id}_ses-clinical01_task-ictal{int(seizure.start)}_*.edf"
                    )
                )
                all_edf_files.extend(edf_files)
            
            # Interictal files
            patient_dir = os.path.join(
                self.config.RAW_DATA_DIR, patient_id, "ses-clinical01", "ieeg"
            )
            if os.path.exists(patient_dir):
                interictal_files = glob(
                    os.path.join(patient_dir, f"{patient_id}_ses-clinical01_task-interictal*_ieeg.edf")
                )
                all_edf_files.extend(interictal_files)
        
        print(f"Found {len(all_edf_files)} total EDF files")
        
        # Randomly sample n_files
        import random
        random.seed(42)  # Reproducibility
        sampled_files = random.sample(all_edf_files, min(n_files, len(all_edf_files)))
        
        print(f"Sampling {len(sampled_files)} files for statistics computation...\n")
        
        # Collect all signal values (vectorized)
        all_values = []
        n_samples_collected = 0
        
        for edf_file in tqdm(sampled_files, desc="Processing files"):
            try:
                # Extract patient ID from filename
                basename = os.path.basename(edf_file)
                patient_id = basename.split('_')[0]
                
                # Preprocess the file
                processed_raw, bipolar_ch_names = self._preprocess_edf(edf_file, patient_id)
                
                if processed_raw is None:
                    continue
                
                # Get all data from this file (before per-channel normalization)
                data = processed_raw.get_data()  # (n_channels, n_samples)
                
                # Flatten and collect (vectorized)
                all_values.append(data.flatten())
                n_samples_collected += data.size
                
            except Exception as e:
                continue
        
        if len(all_values) == 0:
            raise ValueError("No valid data collected from sampled files")
        
        # Concatenate all values (vectorized)
        print(f"\nConcatenating {n_samples_collected:,} samples...")
        all_values = np.concatenate(all_values)
        
        # Compute global statistics (vectorized, ignoring NaNs)
        print(f"Computing global statistics...")
        global_mean = np.nanmean(all_values)
        global_std = np.nanstd(all_values)
        
        stats = {
            'global_mean': float(global_mean),
            'global_std': float(global_std),
            'n_samples_used': int(n_samples_collected),
            'n_files_sampled': len(sampled_files)
        }
        
        print(f"\n✓ Global Statistics Computed:")
        print(f"  Mean: {global_mean:.6f}")
        print(f"  Std: {global_std:.6f}")
        print(f"  Samples: {n_samples_collected:,}")
        print(f"  Files: {len(sampled_files)}")
        
        # Save if output file specified
        if output_file:
            self._save_global_stats(stats, output_file)
        
        return stats
    
    def _match_channels_and_normalize(self, clip, patient_id):
        """
        Match channels to annotations, filter to valid regions, and normalize.
        From make_dataset.py lines 126-161.
        
        Supports both per-channel and global normalization.
        
        Returns:
            tuple: (clip_data, clip_coords, clip_reg_idx, ch_names) or (None, None, None, None)
        """
        try:
            # Step 9: Get corresponding rows of master bipolars (lines 126-128)
            pt_bipolars = self.master_bipolars.loc[
                self.master_bipolars.index == patient_id
            ].copy()
            
            if len(pt_bipolars) == 0:
                return None, None, None, None
            
            # Keep rows that have name in clip.ch_names (lines 136-138)
            pt_bipolars = pt_bipolars.loc[
                pt_bipolars.name.isin(clip.ch_names)
            ].copy()
            
            # Filter to valid regions (lines 139-141)
            pt_bipolars = pt_bipolars.loc[
                pt_bipolars.reg.isin(self.regs.reg)
            ]
            
            if len(pt_bipolars) == 0:
                return None, None, None, None
            
            # Step 10: Apply normalization (line 145)
            if self.use_global_norm:
                # Global normalization: (x - global_mean) / global_std (vectorized)
                clip = clip.apply_function(
                    lambda x: (x - self.global_stats['mean']) / self.global_stats['std'],
                    n_jobs=1
                )
            else:
                # Per-channel z-score normalization (original method)
                clip = clip.apply_function(lambda x: (x - np.nanmean(x)) / np.nanstd(x))
            
            # Make sure clip has same channels as pt_bipolars (line 148)
            # Use pick() instead of legacy pick_channels() to avoid warnings
            clip = clip.pick(pt_bipolars.name.to_list())
            
            # Step 11: Convert to float16 (line 151)
            clip_data = clip.get_data().astype(np.float16)
            clip_coords = pt_bipolars[["mni_x", "mni_y", "mni_z"]].values
            clip_reg = pt_bipolars["reg"].values
            
            ch_names = clip.ch_names
            
            # Filter to valid regions (lines 157-161)
            clip_data = clip_data[np.isin(clip_reg, self.regs.reg.values)]
            clip_coords = clip_coords[np.isin(clip_reg, self.regs.reg.values)]
            clip_reg = clip_reg[np.isin(clip_reg, self.regs.reg.values)]
            clip_reg_idx = np.array([np.where(self.regs.reg.values == i)[0][0] for i in clip_reg])
            ch_names = np.array(ch_names)[np.isin(clip_reg, self.regs.reg.values)]
            
            if len(clip_reg_idx) == 0:
                return None, None, None, None
            
            return clip_data, clip_coords, clip_reg_idx, ch_names
            
        except Exception as e:
            return None, None, None, None
    
    def _extract_windows_from_ictal(self, edf_file, patient_id, seizure_start, seizure_end):
        """
        Extract all non-overlapping 10s windows from an ictal EDF file.
        
        Windows are labeled based on timing:
        - Pre-ictal (0-30s): label=0 (interictal)
        - Ictal (30s to seizure offset): label=1 (ictal)
        - Post-ictal (after offset): label=0 (interictal)
        """
        windows = []
        
        # Preprocess the entire file
        processed_raw, bipolar_ch_names = self._preprocess_edf(edf_file, patient_id)
        
        if processed_raw is None:
            return windows
        
        total_duration = processed_raw.times[-1]
        
        # Determine seizure timing in file (from make_dataset.py lines 107-113)
        try:
            original_raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
            if len(original_raw.annotations) > 0 and original_raw.annotations.onset[0] == 30:
                seizure_start_in_file = 30.0
            else:
                seizure_start_in_file = 0.0
        except:
            seizure_start_in_file = 30.0  # Default assumption
        
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
            
            # Extract and process this window
            try:
                # Crop to window (from make_dataset.py lines 121-123)
                clip = processed_raw.copy().crop(
                    tmin=window_start, tmax=window_end, include_tmax=False, verbose=False
                )
                
                # Check duration (10 seconds = 2560 samples at 256Hz)
                expected_samples = self.window_duration * self.target_sf
                if clip.n_times < expected_samples:
                    window_start += self.window_duration
                    continue
                
                # Match channels and normalize
                clip_data, clip_coords, clip_reg_idx, ch_names = self._match_channels_and_normalize(
                    clip, patient_id
                )
                
                if clip_data is None:
                    window_start += self.window_duration
                    continue
                
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
                    clip_data,  # Already float16
                    clip_coords.astype(np.float32),
                    clip_reg_idx.astype(np.int32),
                    ch_names.tolist() if isinstance(ch_names, np.ndarray) else ch_names,
                    label,
                    metadata
                ))
                
            except Exception as e:
                # Skip this window on error
                pass
            
            window_start += self.window_duration
        
        return windows
    
    def _extract_windows_from_interictal(self, edf_file, patient_id):
        """
        Extract all non-overlapping 10s windows from an interictal EDF file.
        All windows labeled as interictal (0).
        """
        windows = []
        
        # Preprocess the entire file
        processed_raw, bipolar_ch_names = self._preprocess_edf(edf_file, patient_id)
        
        if processed_raw is None:
            return windows
        
        total_duration = processed_raw.times[-1]
        
        # Extract all non-overlapping 10s windows (all labeled as interictal)
        window_start = 0
        while window_start + self.window_duration <= total_duration:
            window_end = window_start + self.window_duration
            
            # Extract and process this window
            try:
                # Crop to window
                clip = processed_raw.copy().crop(
                    tmin=window_start, tmax=window_end, include_tmax=False, verbose=False
                )
                
                # Check duration (10 seconds = 2560 samples at 256Hz)
                expected_samples = self.window_duration * self.target_sf
                if clip.n_times < expected_samples:
                    window_start += self.window_duration
                    continue
                
                # Match channels and normalize
                clip_data, clip_coords, clip_reg_idx, ch_names = self._match_channels_and_normalize(
                    clip, patient_id
                )
                
                if clip_data is None:
                    window_start += self.window_duration
                    continue
                
                # Store window (all interictal)
                metadata = {
                    'file': os.path.basename(edf_file),
                    'window_start': window_start,
                    'window_end': window_end,
                    'label_desc': 'interictal',
                    'file_type': 'interictal'
                }
                
                windows.append((
                    clip_data,  # Already float16
                    clip_coords.astype(np.float32),
                    clip_reg_idx.astype(np.int32),
                    ch_names.tolist() if isinstance(ch_names, np.ndarray) else ch_names,
                    0,  # All interictal
                    metadata
                ))
                
            except Exception as e:
                # Skip this window on error
                pass
            
            window_start += self.window_duration
        
        return windows
    
    def extract_patient(self, patient_id):
        """
        Extract all windows for a single patient from both ictal and interictal files.
        Variable channel counts across files are allowed.
        
        Returns:
            list of tuples: (signals, coords, regs, ch_names, label, metadata)
        """
        all_windows = []
        
        # Process ictal files
        patient_seizures = self.sz_table[self.sz_table['Patient'] == patient_id]
        
        for idx, seizure in patient_seizures.iterrows():
            # Find ictal EDF file
            edf_files = glob(
                os.path.join(
                    self.config.RAW_DATA_DIR,
                    patient_id,
                    "ses-clinical01",
                    "ieeg",
                    f"{patient_id}_ses-clinical01_task-ictal{int(seizure.start)}_*.edf"
                )
            )
            
            if len(edf_files) != 1:
                continue
            
            edf_file = edf_files[0]
            windows = self._extract_windows_from_ictal(
                edf_file, patient_id, seizure.start, seizure.end
            )
            
            all_windows.extend(windows)
        
        # Process interictal files
        patient_dir = os.path.join(
            self.config.RAW_DATA_DIR,
            patient_id,
            "ses-clinical01",
            "ieeg"
        )
        
        if os.path.exists(patient_dir):
            interictal_files = glob(
                os.path.join(patient_dir, f"{patient_id}_ses-clinical01_task-interictal*_ieeg.edf")
            )
            
            for edf_file in interictal_files:
                windows = self._extract_windows_from_interictal(edf_file, patient_id)
                all_windows.extend(windows)
        
        return all_windows
    
    def save_patient_npz(self, patient_id, windows, output_dir):
        """
        Save a single patient's windows to a compressed NPZ file.
        Handles variable channel counts by padding to max_channels.
        
        Args:
            patient_id: Patient identifier
            windows: List of (signals, coords, regs, ch_names, label, metadata) tuples
            output_dir: Directory to save patient NPZ files
        """
        if len(windows) == 0:
            return
        
        # Create output file for this patient
        output_file = os.path.join(output_dir, f"{patient_id}.npz")
        
        n_samples = len(windows)
        n_timesteps = windows[0][0].shape[1]
        
        # Find maximum number of channels across all windows for this patient
        max_channels = max(w[0].shape[0] for w in windows)
        
        # Pre-allocate arrays with padding
        signals_arr = np.zeros((n_samples, max_channels, n_timesteps), dtype=np.float16)
        coords_arr = np.zeros((n_samples, max_channels, 3), dtype=np.float32)
        regs_arr = np.zeros((n_samples, max_channels), dtype=np.int32)
        n_channels_arr = np.zeros(n_samples, dtype=np.int32)
        labels_arr = np.zeros(n_samples, dtype=np.int32)
        
        # Metadata lists
        ch_names_list = []
        window_info_list = []
        
        # Fill arrays (with padding if necessary)
        for i, window in enumerate(windows):
            signals, coords, regs, ch_names, label, metadata = window
            
            n_ch = signals.shape[0]
            n_channels_arr[i] = n_ch
            
            # Store data (padded if necessary)
            signals_arr[i, :n_ch, :] = signals
            coords_arr[i, :n_ch, :] = coords
            regs_arr[i, :n_ch] = regs
            
            labels_arr[i] = label
            ch_names_list.append(json.dumps(ch_names))
            window_info_list.append(json.dumps(metadata))
        
        # Save to compressed NPZ file
        try:
            np.savez_compressed(
                output_file,
                signals=signals_arr,
                coords=coords_arr,
                regs=regs_arr,
                n_channels=n_channels_arr,
                labels=labels_arr,
                patient_id=patient_id,
                ch_names=np.array(ch_names_list, dtype=object),
                window_info=np.array(window_info_list, dtype=object)
            )
            
            # Count ictal vs interictal
            n_ictal = np.sum(labels_arr == 1)
            n_interictal = np.sum(labels_arr == 0)
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            
            print(f"  ✓ Saved {patient_id}.npz:")
            print(f"    - Total windows: {n_samples}")
            print(f"    - Ictal: {n_ictal}, Interictal: {n_interictal}")
            print(f"    - Max channels: {max_channels}")
            print(f"    - File size: {file_size_mb:.1f} MB")
        except Exception as e:
            print(f"  ✗ Error saving {patient_id}.npz: {e}")
    
    def extract_all_to_directory(self, output_dir, test_patient=None):
        """
        Extract all windows for all patients (or single test patient) and save to NPZ files.
        
        Args:
            output_dir: Directory to save per-patient NPZ files
            test_patient: If provided, only process this one patient (for testing)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get unique patients
        if test_patient:
            unique_patients = [test_patient]
            print(f"\n*** TEST MODE: Processing single patient {test_patient} ***\n")
        else:
            unique_patients = self.sz_table['Patient'].unique()
        
        print(f"Extracting windows from {len(unique_patients)} patient(s)...")
        print(f"Saving to directory: {output_dir}")
        print(f"Format: One compressed NPZ file per patient\n")
        
        total_windows = 0
        total_interictal = 0
        total_ictal = 0
        patients_processed = 0
        
        for patient_id in tqdm(unique_patients, desc="Patients"):
            # print(f"\nProcessing {patient_id}...")
            
            # Extract windows for this patient
            windows = self.extract_patient(patient_id)
            
            if len(windows) == 0:
                print(f"  No windows extracted for {patient_id}")
                continue
            
            # Save immediately (frees memory)
            self.save_patient_npz(patient_id, windows, output_dir)
            
            # Count statistics
            n_interictal = sum(1 for w in windows if w[4] == 0)
            n_ictal = sum(1 for w in windows if w[4] == 1)
            
            total_windows += len(windows)
            total_interictal += n_interictal
            total_ictal += n_ictal
            patients_processed += 1
            
            # Windows are now freed from memory
        
        print(f"\n{'='*60}")
        print(f"✓ Extraction complete!")
        print(f"  Patients processed: {patients_processed}")
        print(f"  Total windows: {total_windows}")
        print(f"  Interictal (label=0): {total_interictal}")
        print(f"  Ictal (label=1): {total_ictal}")
        print(f"  Files saved: {output_dir}/*.npz")
        print(f"{'='*60}")


def main():
    """Extract all windows from all patients (or test on single patient)."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract all 10-second windows from iEEG data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute global statistics (run once)
  python extract_all_windows.py --compute-global-stats

  # Extract with per-channel normalization (default)
  python extract_all_windows.py
  
  # Extract with global normalization
  python extract_all_windows.py --use-global-norm
  
  # Test on single patient
  python extract_all_windows.py --test-patient sub-RID0106
        """
    )
    parser.add_argument(
        '--test-patient',
        type=str,
        default=None,
        help='Test on a single patient (e.g., sub-RID0106)'
    )
    parser.add_argument(
        '--compute-global-stats',
        action='store_true',
        help='Compute global mean/std from 200 random EDF files and save to file (run once)'
    )
    parser.add_argument(
        '--n-files-for-stats',
        type=int,
        default=200,
        help='Number of EDF files to sample for global statistics (default: 200)'
    )
    parser.add_argument(
        '--use-global-norm',
        action='store_true',
        help='Use global normalization instead of per-channel normalization'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='',
        help='Suffix for output directory (e.g., "_global_norm")'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Extract All Windows - All Patients")
    print("Using make_dataset.py preprocessing pipeline")
    print("Saving as compressed NPZ files")
    print("="*60)
    
    # Create config and directories
    config = BenchmarkConfig()
    config.create_directories()
    
    # Global statistics file path
    global_stats_file = os.path.join(
        config.OUTPUT_DATA_DIR,
        "global_normalization_stats.json"
    )
    
    # Mode 1: Compute global statistics
    if args.compute_global_stats:
        print("\n*** MODE: Computing Global Statistics ***\n")
        extractor = AllWindowsExtractor(config, use_global_norm=False)
        extractor.compute_global_stats(
            n_files=args.n_files_for_stats,
            output_file=global_stats_file
        )
        print(f"\n✓ Statistics saved to: {global_stats_file}")
        print(f"\nTo extract windows with global normalization, run:")
        print(f"  python extract_all_windows.py --use-global-norm")
        return
    
    # Mode 2: Extract windows
    print(f"\n*** MODE: Extracting Windows ***")
    print(f"Normalization: {'Global' if args.use_global_norm else 'Per-channel'}\n")
    
    # Create extractor
    extractor = AllWindowsExtractor(
        config,
        use_global_norm=args.use_global_norm,
        global_stats_file=global_stats_file if args.use_global_norm else None
    )
    
    # Output directory for per-patient NPZ files
    output_dir_name = "all_windows_per_patient"
    if args.output_suffix:
        output_dir_name += args.output_suffix
    elif args.use_global_norm:
        output_dir_name += "_global_norm"
    
    output_dir = os.path.join(config.OUTPUT_DATA_DIR, output_dir_name)
    
    print(f"Output directory: {output_dir}\n")
    
    # Extract and save per-patient (or single test patient)
    extractor.extract_all_to_directory(output_dir, test_patient=args.test_patient)
    
    print(f"\n✓ Per-patient NPZ files saved to: {output_dir}")


if __name__ == "__main__":
    main()
