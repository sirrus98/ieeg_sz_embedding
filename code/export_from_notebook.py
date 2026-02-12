"""
Helper code to export embeddings from within the Jupyter notebook
Copy and paste this into a new cell in test_nb.ipynb after Cell 17 (where sz_embeddings is created)
"""

# Run this cell after Cell 17 in test_nb.ipynb to export embeddings

import pandas as pd
import numpy as np
from os.path import join as ospj
from config import CONFIG

# Create export dataframe
print("Creating export dataframe...")
export_df = pd.DataFrame()

# Add RID (patient ID)
export_df['RID'] = sz_metadata['Patient'].values

# Add other metadata if available
if 'IEEGID' in sz_metadata.columns:
    export_df['IEEGID'] = sz_metadata['IEEGID'].values
if 'IEEGname' in sz_metadata.columns:
    export_df['IEEGname'] = sz_metadata['IEEGname'].values
if 'start' in sz_metadata.columns:
    export_df['start'] = sz_metadata['start'].values
if 'end' in sz_metadata.columns:
    export_df['end'] = sz_metadata['end'].values
if 'source' in sz_metadata.columns:
    export_df['source'] = sz_metadata['source'].values
if 'Semiology' in sz_metadata.columns:
    export_df['Semiology'] = sz_metadata['Semiology'].values

# Add embedding dimensions
print(f"Adding {sz_embeddings.shape[1]} embedding dimensions...")
for i in range(sz_embeddings.shape[1]):
    export_df[f'embedding_{i}'] = sz_embeddings[:, i]

# Save to CSV - use user's writable directory
import os
output_dir = "/users/zcxu/ieeg_sz_embedding/data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
output_path = ospj(output_dir, "seizure_embeddings_with_rid.csv")
export_df.to_csv(output_path, index=False)
print(f"✓ Exported to: {output_path}")
print(f"  Shape: {export_df.shape}")

# Also save as numpy array for faster loading
np_output_path = output_path.replace('.csv', '.npy')
np.save(np_output_path, sz_embeddings)
print(f"✓ Saved numpy array to: {np_output_path}")

# Save metadata separately
metadata_output_path = output_path.replace('.csv', '_metadata.csv')
sz_metadata.to_csv(metadata_output_path, index=False)
print(f"✓ Saved metadata to: {metadata_output_path}")

# Print summary
print("\n=== Summary ===")
print(f"Total embeddings: {len(sz_embeddings)}")
print(f"Unique patients (RID): {export_df['RID'].nunique()}")
print(f"Embedding dimension: {sz_embeddings.shape[1]}")
print("\nSeizures per patient:")
print(export_df['RID'].value_counts().head(10))
print("\n✓ Export completed!")
