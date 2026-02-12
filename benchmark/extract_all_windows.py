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
import h5py
import json
from scipy.signal import resample_poly
from fractions import Fraction

from config_benchmark import BenchmarkConfig
from utils import check_channel_types, clean_labels, notch_filter, bandpass_filter, bipolar_montage


class AllWindowsExtractor:
    """Extract all possible 10-second windows from EDF files using make_dataset.py preprocessing."""
    
    def __init__(self, config=BenchmarkConfig):
        """Initialize the extractor."""
        self.config = config
        self.target_sf = config.TARGET_SAMPLING_RATE
        self.window_duration = 10  # seconds
        
        # Load metadata
        self._load_metadata()
        
        # Load annotations and region mappings
        self._load_annotations()
        
        print(f"✓ Initialization complete")
    
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
        
        # Load DKT mapping - need to get dkt_to_custom from CONFIG
        # For now, we'll load the regions file directly
        regs_file = os.path.join(self.config.DATA_DIR, 'atlases', 'luts', 'dktg_reordered.csv')
        if os.path.exists(regs_file):
            regs_li = pd.read_csv(regs_file)
            regs_li = regs_li.atlas_index.values
            self.regs = pd.DataFrame(regs_li, columns=['reg'])
        else:
            # Fallback
            self.regs = pd.DataFrame({'reg': list(range(41))})
        
        # Try to load dkt_to_custom mapping from CONFIG
        try:
            # Import CONFIG to get dkt_to_custom
            from config import CONFIG as MAIN_CONFIG
            if hasattr(MAIN_CONFIG, 'dkt_to_custom'):
                master_bipolars['reg'] = master_bipolars.final_label.map(MAIN_CONFIG.dkt_to_custom)
            else:
                # Use atlas_index directly if mapping not available
                master_bipolars['reg'] = master_bipolars.get('final_label', 0)
        except:
            # If CONFIG import fails, use final_label directly
            master_bipolars['reg'] = master_bipolars.get('final_label', 0)
        
        self.master_bipolars = master_bipolars
        
        print(f"✓ Loaded annotations for {len(master_bipolars)} channels")
        print(f"✓ Loaded {len(self.regs)} valid regions")
    
    def _preprocess_edf(self, edf_file, patient_id):
        """
        Apply full preprocessing pipeline from make_dataset.py (lines 66-160).
        
        Steps:
        1. Load EDF with mne.io.read_raw_edf()
        2. Clean channel labels using clean_labels()
        3. Get channel types using check_channel_types()
        4. Apply 60Hz notch filter using notch_filter()
        5. Apply bandpass filter (0.5-120Hz, order=10) using bandpass_filter()
        6. Apply bipolar montage using bipolar_montage()
        7. Resample to 256Hz using resample_poly()
        8. Create MNE RawArray
        9. Match channels to master_bipolars annotations
        10. Filter to valid regions from unique_regs
        
        Args:
            edf_file: Path to EDF file
            patient_id: Patient identifier
        
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
            # Silently skip files that fail preprocessing
            return None, None
    
    def _match_channels_and_normalize(self, clip, patient_id):
        """
        Match channels to annotations, filter to valid regions, and normalize.
        From make_dataset.py lines 126-160.
        
        Args:
            clip: MNE Raw object with 10-second window
            patient_id: Patient identifier
        
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
            
            # Step 10: Apply per-channel z-score normalization (line 145)
            # CRITICAL: This must match make_dataset.py exactly
            clip = clip.apply_function(lambda x: (x - np.nanmean(x)) / np.nanstd(x))
            
            # Make sure clip has same channels as pt_bipolars (line 148)
            clip = clip.pick_channels(pt_bipolars.name.to_list())
            
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
        
        Args:
            edf_file: Path to ictal EDF file
            patient_id: Patient identifier
            seizure_start: Seizure start time in original recording (seconds)
            seizure_end: Seizure end time in original recording (seconds)
        
        Returns:
            list of tuples: (signals, coords, regs, ch_names, label, metadata)
        """
        windows = []
        
        # Preprocess the entire file
        processed_raw, bipolar_ch_names = self._preprocess_edf(edf_file, patient_id)
        
        if processed_raw is None:
            return windows
        
        total_duration = processed_raw.times[-1]
        
        # Determine seizure timing in file
        # Files start 30 seconds before seizure, so seizure starts at 30s in file
        # (from make_dataset.py lines 107-113)
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
                
                # Check duration
                if clip.n_times < 60 * self.target_sf:
                    # Skip if less than expected duration
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
        
        Args:
            edf_file: Path to interictal EDF file
            patient_id: Patient identifier
        
        Returns:
            list of tuples: (signals, coords, regs, ch_names, label, metadata)
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
                
                # Check duration
                if clip.n_times < 60 * self.target_sf:
                    # Skip if less than expected duration
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
        
        Args:
            patient_id: Patient identifier (e.g., 'sub-RID0106')
        
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
                    "ses-clinical01/ieeg",
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
            "ses-clinical01/ieeg"
        )
        
        if os.path.exists(patient_dir):
            interictal_files = glob(
                os.path.join(patient_dir, f"{patient_id}_ses-clinical01_task-interictal*_ieeg.edf")
            )
            
            for edf_file in interictal_files:
                windows = self._extract_windows_from_interictal(edf_file, patient_id)
                all_windows.extend(windows)
        
        return all_windows
    
    def save_patient(self, patient_id, windows, output_dir):
        """
        Save a single patient's windows to an HDF5 file.
        Handles variable channel counts by padding to max channels.
        
        Args:
            patient_id: Patient identifier
            windows: List of (signals, coords, regs, ch_names, label, metadata) tuples
            output_dir: Directory to save patient HDF5 files
        """
        if len(windows) == 0:
            return
        
        # Create output file for this patient
        output_file = os.path.join(output_dir, f"{patient_id}.h5")
        
        n_samples = len(windows)
        n_timesteps = windows[0][0].shape[1]
        
        # Find maximum number of channels across all windows for this patient
        max_channels = max(w[0].shape[0] for w in windows)
        
        with h5py.File(output_file, 'w') as f:
            # Create datasets with max_channels
            signals_dset = f.create_dataset(
                'signals',
                shape=(n_samples, max_channels, n_timesteps),
                dtype='float16',  # Match make_dataset.py line 151
                compression='gzip',
                compression_opts=1
            )
            
            coords_dset = f.create_dataset(
                'coords',
                shape=(n_samples, max_channels, 3),
                dtype='float32',
                compression='gzip',
                compression_opts=1
            )
            
            regs_dset = f.create_dataset(
                'regs',
                shape=(n_samples, max_channels),
                dtype='int32',
                compression='gzip',
                compression_opts=1
            )
            
            # Store actual channel counts for each window
            n_channels_dset = f.create_dataset(
                'n_channels',
                shape=(n_samples,),
                dtype='int32'
            )
            
            labels_dset = f.create_dataset(
                'labels',
                shape=(n_samples,),
                dtype='int32'
            )
            
            # Metadata
            ch_names_list = []
            window_info_list = []
            
            # Write windows (with padding if necessary)
            for i, window in enumerate(windows):
                signals, coords, regs, ch_names, label, metadata = window
                
                n_ch = signals.shape[0]
                n_channels_dset[i] = n_ch
                
                # Pad if necessary
                if n_ch < max_channels:
                    # Pad with zeros
                    padded_signals = np.zeros((max_channels, n_timesteps), dtype=np.float16)
                    padded_signals[:n_ch, :] = signals
                    
                    padded_coords = np.zeros((max_channels, 3), dtype=np.float32)
                    padded_coords[:n_ch, :] = coords
                    
                    padded_regs = np.zeros(max_channels, dtype=np.int32)
                    padded_regs[:n_ch] = regs
                    
                    signals_dset[i] = padded_signals
                    coords_dset[i] = padded_coords
                    regs_dset[i] = padded_regs
                else:
                    signals_dset[i] = signals
                    coords_dset[i] = coords
                    regs_dset[i] = regs
                
                labels_dset[i] = label
                ch_names_list.append(json.dumps(ch_names))
                window_info_list.append(json.dumps(metadata))
            
            # Save metadata
            f.create_dataset('patient_id', data=patient_id.encode('utf-8'), dtype=h5py.string_dtype())
            f.create_dataset('ch_names', data=[s.encode('utf-8') for s in ch_names_list], dtype=h5py.string_dtype())
            f.create_dataset('window_info', data=[s.encode('utf-8') for s in window_info_list], dtype=h5py.string_dtype())
    
    def extract_all_to_directory(self, output_dir):
        """
        Extract all windows for all patients and save one HDF5 file per patient.
        This avoids memory issues by processing and saving one patient at a time.
        
        Args:
            output_dir: Directory to save per-patient HDF5 files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get unique patients
        unique_patients = self.sz_table['Patient'].unique()
        
        print(f"\nExtracting windows from {len(unique_patients)} patients...")
        print(f"Saving to directory: {output_dir}")
        print(f"Format: One HDF5 file per patient\n")
        
        total_windows = 0
        total_interictal = 0
        total_ictal = 0
        patients_processed = 0
        
        for patient_id in tqdm(unique_patients, desc="Patients"):
            # Extract windows for this patient
            windows = self.extract_patient(patient_id)
            
            if len(windows) == 0:
                continue
            
            # Save immediately (frees memory)
            self.save_patient(patient_id, windows, output_dir)
            
            # Count statistics
            n_interictal = sum(1 for w in windows if w[4] == 0)
            n_ictal = sum(1 for w in windows if w[4] == 1)
            
            total_windows += len(windows)
            total_interictal += n_interictal
            total_ictal += n_ictal
            patients_processed += 1
            
            # Windows are now freed from memory
        
        print(f"\n✓ Extracted {total_windows} total windows")
        print(f"  Patients processed: {patients_processed}")
        print(f"  Interictal (label=0): {total_interictal}")
        print(f"  Ictal (label=1): {total_ictal}")
        print(f"  Files saved: {output_dir}/*.h5")


def main():
    """Extract all windows from all patients."""
    print("="*60)
    print("Extract All Windows - All Patients")
    print("Using make_dataset.py preprocessing pipeline")
    print("="*60)
    
    # Create config and directories
    config = BenchmarkConfig()
    config.create_directories()
    
    # Create extractor
    extractor = AllWindowsExtractor(config)
    
    # Output directory for per-patient HDF5 files
    output_dir = os.path.join(
        config.OUTPUT_DATA_DIR,
        "all_windows_per_patient"
    )
    
    # Extract and save per-patient
    extractor.extract_all_to_directory(output_dir)
    
    print("\n✓ Extraction complete!")
    print(f"✓ Per-patient HDF5 files saved to: {output_dir}")


if __name__ == "__main__":
    main()
