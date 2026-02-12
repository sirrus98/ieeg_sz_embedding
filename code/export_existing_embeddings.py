"""
Export embeddings that were already generated in the notebook
This script loads pre-computed embeddings and exports them with patient IDs
"""

import os
from os.path import join as ospj
import numpy as np
import pandas as pd

from config import CONFIG


def export_embeddings_from_notebook():
    """
    Export embeddings that were already computed in the notebook
    Loads the numpy arrays and metadata that were saved during notebook execution
    """
    print("Looking for pre-computed embeddings from notebook...")
    
    # Try to load embeddings from various possible locations
    possible_paths = [
        # From the notebook, embeddings might be saved in gui_data
        ospj(CONFIG.data_dir, "gui_data", "sz_embeddings.npy"),
        ospj(CONFIG.data_dir, "sz_embeddings.npy"),
        # Or they might still be in memory from the notebook
    ]
    
    embeddings = None
    embeddings_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found embeddings at: {path}")
            embeddings = np.load(path)
            embeddings_path = path
            break
    
    if embeddings is None:
        print("Pre-computed embeddings not found. You need to run generate_embeddings.py first.")
        print("Or generate them in the notebook and save with:")
        print("  np.save(ospj(CONFIG.data_dir, 'gui_data', 'sz_embeddings.npy'), sz_embeddings)")
        return False
    
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # Try to load metadata
    possible_metadata_paths = [
        ospj(CONFIG.data_dir, "gui_data", "master_metadata.csv"),
        ospj(CONFIG.data_dir, "master_metadata.csv"),
    ]
    
    metadata = None
    for path in possible_metadata_paths:
        if os.path.exists(path):
            print(f"Found metadata at: {path}")
            metadata = pd.read_csv(path)
            break
    
    if metadata is None:
        print("Metadata not found. Creating minimal export with just embeddings...")
        # Create minimal metadata
        metadata = pd.DataFrame({
            'Patient': [f'sample_{i}' for i in range(len(embeddings))]
        })
    
    # Filter metadata to match embeddings length
    if len(metadata) > len(embeddings):
        print(f"Truncating metadata from {len(metadata)} to {len(embeddings)} rows")
        metadata = metadata.iloc[:len(embeddings)]
    elif len(metadata) < len(embeddings):
        print(f"Warning: Metadata has {len(metadata)} rows but embeddings has {len(embeddings)}")
        print("Using available metadata and padding with unknown for extra rows")
        extra_rows = len(embeddings) - len(metadata)
        extra_df = pd.DataFrame({
            'Patient': [f'unknown_{i}' for i in range(extra_rows)]
        })
        metadata = pd.concat([metadata, extra_df], ignore_index=True)
    
    # Create export dataframe
    export_df = pd.DataFrame()
    
    # Add RID (patient ID)
    if 'Patient' in metadata.columns:
        export_df['RID'] = metadata['Patient'].values
    elif 'rid' in metadata.columns:
        export_df['RID'] = metadata['rid'].values
    else:
        export_df['RID'] = [f'sample_{i}' for i in range(len(embeddings))]
    
    # Add other metadata columns if available
    metadata_cols = ['IEEGID', 'IEEGname', 'start', 'end', 'source', 'notes', 'Semiology']
    for col in metadata_cols:
        if col in metadata.columns:
            export_df[col] = metadata[col].values
    
    # Add embedding dimensions
    for i in range(embeddings.shape[1]):
        export_df[f'embedding_{i}'] = embeddings[:, i]
    
    # Save to CSV
    output_path = ospj(CONFIG.data_dir, "seizure_embeddings_with_rid.csv")
    export_df.to_csv(output_path, index=False)
    print(f"\n✓ Exported embeddings to: {output_path}")
    print(f"  Shape: {export_df.shape}")
    print(f"  Columns: {list(export_df.columns[:5])}... + {embeddings.shape[1]} embedding dimensions")
    
    # Save embeddings as numpy array
    np_output_path = output_path.replace('.csv', '.npy')
    np.save(np_output_path, embeddings)
    print(f"✓ Saved embeddings as numpy array to: {np_output_path}")
    
    # Save metadata separately
    metadata_output_path = output_path.replace('.csv', '_metadata.csv')
    metadata.to_csv(metadata_output_path, index=False)
    print(f"✓ Saved metadata to: {metadata_output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total embeddings: {len(embeddings)}")
    if 'RID' in export_df.columns:
        n_patients = export_df['RID'].nunique()
        print(f"Unique patients: {n_patients}")
        print("\nTop 10 patients by number of seizures:")
        print(export_df['RID'].value_counts().head(10))
    
    return True


def main():
    """Main function"""
    print("=" * 60)
    print("Export Existing Embeddings")
    print("=" * 60)
    print()
    
    success = export_embeddings_from_notebook()
    
    if success:
        print("\n✓ Export completed successfully!")
        print("\nNext steps:")
        print("  1. Load in Python: embeddings = np.load('seizure_embeddings_with_rid.npy')")
        print("  2. Load in Python: df = pd.read_csv('seizure_embeddings_with_rid.csv')")
        print("  3. Or use your preferred analysis tool")
    else:
        print("\n✗ Export failed. Run generate_embeddings.py to create embeddings first.")


if __name__ == "__main__":
    main()
