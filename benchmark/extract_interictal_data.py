"""
Extract interictal (non-seizure) data for ictal vs interictal classification.

This script extracts 10-second windows from dedicated interictal EDF files,
applying the same preprocessing as ictal data. Windows are randomly sampled
from separate task-interictal recordings to ensure true interictal data.
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
    
    def _find_interictal_files(self, patient_id):
        """
        Find interictal EDF files for a patient.
        
        Args:
            patient_id: Patient identifier (e.g., 'sub-RID0106')
        
        Returns:
            list of interictal EDF file paths
        """
        # Find all interictal files for this patient
        patient_dir = os.path.join(
            self.config.RAW_DATA_DIR,
            patient_id,
            "ses-clinical01/ieeg"
        )
        
        if not os.path.exists(patient_dir):
            return []
        
        # Get all interictal EDF files
        interictal_files = glob(
            os.path.join(patient_dir, f"{patient_id}_ses-clinical01_task-interictal*_ieeg.edf")
            )
            
        return interictal_files
    
    def _create_bipolar_channels(self, data, ch_names):
        """
        Convert monopolar channels to bipolar by taking differences between adjacent channels.
        
        Args:
            data: numpy array of shape (channels, samples)
            ch_names: list of channel names
        
        Returns:
            tuple: (bipolar_data, bipolar_ch_names)
        """
        # Group channels by electrode (e.g., LAST01, LAST02 -> LAST)
        from collections import defaultdict
        electrode_groups = defaultdict(list)
        
        for idx, ch_name in enumerate(ch_names):
            # Extract electrode name (letters before numbers)
            # E.g., "LAST01" -> "LAST", "LDA02" -> "LDA"
            import re
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
            
            # Convert monopolar to bipolar
            bipolar_data, bipolar_names = self._create_bipolar_channels(data, ch_names)
            
            if bipolar_data is None:
                return None
            
            # Get channel coordinates and regions for bipolar channels
            coords = []
            regs = []
            valid_channels = []
            
            patient_annots = self.annotations.loc[patient_id]
            
            for bipolar_name in bipolar_names:
                # Look up in annotations
                if bipolar_name in patient_annots['name'].values:
                    ann = patient_annots[patient_annots['name'] == bipolar_name].iloc[0]
                    if 'mni_x' in ann and 'mni_y' in ann and 'mni_z' in ann:
                        coords.append([ann['mni_x'], ann['mni_y'], ann['mni_z']])
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
            
            bipolar_data = bipolar_data[valid_channels]
            coords = np.array(coords)
            regs = np.array(regs)
            bipolar_names = [bipolar_names[i] for i in range(len(bipolar_names)) if valid_channels[i]]
            
            # Convert to numpy
            signals = bipolar_data.astype(np.float32)
            
            return signals, coords, regs, bipolar_names
        
        except Exception as e:
            # Silently fail for most files
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
            # Find interictal files for this patient
            interictal_files = self._find_interictal_files(patient_id)
            
            if len(interictal_files) == 0:
                continue
            
            # Extract windows from randomly selected files
            patient_windows = []
            
            # Randomly shuffle files to get diverse samples
            np.random.shuffle(interictal_files)
            
            for edf_file in interictal_files:
                # Check file duration
                try:
                    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
                    total_duration = raw.times[-1]
                except:
                    continue
                
                # Check if file is long enough for a window
                if total_duration < self.window_duration:
                    continue
                
                # Extract one window from this file (randomly placed)
                max_start = total_duration - self.window_duration
                start_time = np.random.uniform(0, max_start)
                
                result = self._extract_window(edf_file, start_time, patient_id)
                if result is not None:
                    patient_windows.append(result)
                    
                # Stop if we have enough windows for this patient
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


def test_write_permissions(output_path):
    """
    Test write permissions by creating a temporary test file.
    
    Args:
        output_path: Path where we want to save data
        
    Raises:
        PermissionError: If we don't have write permissions
    """
    test_file = output_path + '.permission_test'
    try:
        print(f"\nTesting write permissions for: {output_path}")
        with open(test_file, 'w') as f:
            f.write("permission test")
        os.remove(test_file)
        print("✓ Write permissions confirmed!")
    except PermissionError as e:
        print(f"\n✗ ERROR: No write permission for {output_path}")
        print(f"  {str(e)}")
        print("\nPlease check that:")
        print(f"  1. The directory exists and you have write access")
        print(f"  2. The OUTPUT_DATA_DIR in config_benchmark.py points to your directory")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: Could not test write permissions: {str(e)}")
        raise


def main():
    """Main function to extract interictal data."""
    print("=" * 60)
    print("Interictal Data Extraction")
    print("=" * 60)
    
    # Create config and directories
    config = BenchmarkConfig()
    config.create_directories()
    
    # Test write permissions BEFORE doing expensive extraction
    test_write_permissions(config.INTERICTAL_DATA_FILE)
    
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
