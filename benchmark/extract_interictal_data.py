"""
Extract interictal (non-seizure) data for ictal vs interictal classification.

This script extracts 10-second windows from interictal periods,
applying the same preprocessing as ictal data.
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
from scipy.signal import resample_poly
from fractions import Fraction

from config_benchmark import BenchmarkConfig
from utils import check_channel_types, clean_labels


class InterictalExtractor:
    """Extract interictal windows from EDF files."""
    
    def __init__(self, config=BenchmarkConfig):
        """
        Initialize the extractor.
        
        Args:
            config: Configuration class with paths and parameters
        """
        self.config = config
        self.target_sf = config.TARGET_SAMPLING_RATE
        self.window_duration = config.INTERICTAL_WINDOW_DURATION
        self.buffer_time = config.INTERICTAL_BUFFER_TIME
        self.windows_per_patient = config.INTERICTAL_WINDOWS_PER_PATIENT
        
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
        
        for ind, row in rid_hup_table.iterrows():
            rid_hup_table.loc[ind, "hupsubjno"] = int(row["hupsubjno"][:3])
        
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
            # Load DKT to custom region mapping (placeholder - needs actual mapping)
            # For now, we'll use the atlas_index directly
            dkt_regs = pd.read_csv(self.config.DKT_MAPPING_FILE)
            # Create a simple mapping (this should be customized based on actual data)
            dkt_to_custom = dict(zip(dkt_regs['atlas_index'], dkt_regs['atlas_index']))
        else:
            # Use identity mapping if file doesn't exist
            dkt_to_custom = {}
        
        if 'final_label' in master_bipolars.columns:
            master_bipolars['reg'] = master_bipolars.final_label.map(
                lambda x: dkt_to_custom.get(x, x) if pd.notna(x) else 0
            )
        else:
            # If no final_label column, set reg to 0
            master_bipolars['reg'] = 0
        
        self.annotations = master_bipolars
        
        print(f"✓ Loaded annotations for {len(master_bipolars)} channels")
    
    def _find_interictal_periods(self, patient_id):
        """
        Find interictal time periods for a patient.
        
        Args:
            patient_id: Patient identifier (e.g., 'sub-RID0106')
        
        Returns:
            list of (edf_file, start_time, end_time) tuples
        """
        # Get all seizures for this patient
        patient_seizures = self.sz_table[self.sz_table['Patient'] == patient_id]
        
        if len(patient_seizures) == 0:
            return []
        
        # For each seizure file, find interictal periods
        interictal_periods = []
        
        for _, seizure in patient_seizures.iterrows():
            # Find the EDF file
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
            
            # Load EDF to get duration
            try:
                raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
                total_duration = raw.times[-1]
            except:
                continue
            
            # Seizure starts at 30 seconds in the file (based on annotation)
            seizure_start_in_file = 30.0
            seizure_duration = seizure.end - seizure.start
            seizure_end_in_file = seizure_start_in_file + seizure_duration
            
            # Find interictal period before seizure (with buffer)
            pre_seizure_end = seizure_start_in_file - self.buffer_time
            if pre_seizure_end > self.window_duration:
                # We have space for at least one window before seizure
                interictal_periods.append((edf_file, 0, pre_seizure_end))
            
            # Find interictal period after seizure (with buffer)
            post_seizure_start = seizure_end_in_file + self.buffer_time
            if post_seizure_start + self.window_duration < total_duration:
                # We have space for at least one window after seizure
                interictal_periods.append((edf_file, post_seizure_start, total_duration))
        
        return interictal_periods
    
    def _extract_window(self, edf_file, start_time, patient_id):
        """
        Extract a single 10-second interictal window.
        
        Args:
            edf_file: Path to EDF file
            start_time: Start time in seconds
            patient_id: Patient identifier
        
        Returns:
            tuple: (signals, coords, regs, ch_names) or None if extraction fails
        """
        try:
            # Load EDF
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            # Crop to window
            end_time = start_time + self.window_duration
            raw_cropped = raw.copy().crop(tmin=start_time, tmax=end_time)
            
            # Get sampling rate
            fs = raw_cropped.info['sfreq']
            
            # Get channel names
            ch_names = raw_cropped.ch_names
            
            # Get data
            data = raw_cropped.get_data()  # (channels, samples)
            
            # Resample to target frequency
            if fs != self.target_sf:
                frac = Fraction(int(self.target_sf), int(fs))
                data = resample_poly(data, frac.numerator, frac.denominator, axis=1)
            
            # Get channel coordinates and regions
            coords = []
            regs = []
            valid_channels = []
            
            for ch_name in ch_names:
                # Look up in annotations
                ch_key = f"{patient_id}_{ch_name}"
                if ch_key in self.annotations.index:
                    ann = self.annotations.loc[ch_key]
                    if 'x' in ann and 'y' in ann and 'z' in ann:
                        coords.append([ann['x'], ann['y'], ann['z']])
                        regs.append(ann.get('reg', 0))
                        valid_channels.append(True)
                    else:
                        valid_channels.append(False)
                else:
                    valid_channels.append(False)
            
            # Filter to valid channels only
            valid_channels = np.array(valid_channels)
            if valid_channels.sum() == 0:
                return None
            
            data = data[valid_channels]
            coords = np.array(coords)
            regs = np.array(regs)
            ch_names = [ch_names[i] for i in range(len(ch_names)) if valid_channels[i]]
            
            # Convert to numpy
            signals = data.astype(np.float32)
            
            return signals, coords, regs, ch_names
        
        except Exception as e:
            print(f"  Warning: Failed to extract window from {edf_file}: {e}")
            return None
    
    def extract_all(self):
        """
        Extract interictal windows for all patients.
        
        Returns:
            tuple: (signals, coords, regs, ch_names, patient_indices)
        """
        all_signals = []
        all_coords = []
        all_regs = []
        all_ch_names = []
        all_patient_indices = []
        
        # Get unique patients
        unique_patients = self.sz_table['Patient'].unique()
        
        print(f"\nExtracting interictal windows from {len(unique_patients)} patients...")
        
        for patient_id in tqdm(unique_patients):
            # Find interictal periods
            periods = self._find_interictal_periods(patient_id)
            
            if len(periods) == 0:
                continue
            
            # Extract windows from these periods
            patient_windows = []
            for edf_file, period_start, period_end in periods:
                # Sample random time points within this period
                available_duration = period_end - period_start - self.window_duration
                if available_duration <= 0:
                    continue
                
                # Sample window start times
                n_samples = min(self.windows_per_patient // len(periods) + 1, 
                               int(available_duration / self.window_duration))
                
                if n_samples == 0:
                    continue
                
                start_times = np.random.uniform(
                    period_start, 
                    period_end - self.window_duration, 
                    size=n_samples
                )
                
                for start_time in start_times:
                    result = self._extract_window(edf_file, start_time, patient_id)
                    if result is not None:
                        patient_windows.append(result)
                    
                    if len(patient_windows) >= self.windows_per_patient:
                        break
                
                if len(patient_windows) >= self.windows_per_patient:
                    break
            
            # Add to collection
            for signals, coords, regs, ch_names in patient_windows:
                all_signals.append(signals)
                all_coords.append(coords)
                all_regs.append(regs)
                all_ch_names.append(ch_names)
                all_patient_indices.append(patient_id)
        
        print(f"\n✓ Extracted {len(all_signals)} interictal windows")
        print(f"  - From {len(set(all_patient_indices))} patients")
        
        return all_signals, all_coords, all_regs, all_ch_names, all_patient_indices
    
    def save(self, signals, coords, regs, ch_names, patient_indices):
        """
        Save extracted data to pickle file.
        
        Args:
            signals: List of signal arrays
            coords: List of coordinate arrays
            regs: List of region arrays
            ch_names: List of channel name lists
            patient_indices: List of patient IDs
        """
        output_file = self.config.INTERICTAL_DATA_FILE
        
        # Save to pickle
        with open(output_file, 'wb') as f:
            pickle.dump((signals, coords, regs, ch_names, patient_indices), f)
        
        print(f"\n✓ Saved interictal data to: {output_file}")


def main():
    """Main function to extract interictal data."""
    print("=" * 60)
    print("Interictal Data Extraction")
    print("=" * 60)
    
    # Create config and directories
    config = BenchmarkConfig()
    config.create_directories()
    
    # Create extractor
    extractor = InterictalExtractor(config)
    
    # Extract data
    signals, coords, regs, ch_names, patient_indices = extractor.extract_all()
    
    # Save
    extractor.save(signals, coords, regs, ch_names, patient_indices)
    
    print("\n✓ Interictal extraction complete!")
    print(f"  Total windows: {len(signals)}")
    print(f"  Patients: {len(set(patient_indices))}")


if __name__ == "__main__":
    main()
