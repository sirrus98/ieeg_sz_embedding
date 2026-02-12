"""
Quick script to verify the exported embeddings
"""

import pandas as pd
import numpy as np
from os.path import join as ospj

# Load the CSV file
print("Loading exported embeddings...")
df = pd.read_csv('/users/zcxu/ieeg_sz_embedding/data/seizure_embeddings_with_rid.csv')

print(f"\n✓ Successfully loaded CSV file")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns[:5])} + {df.shape[1]-5} more columns")

# Extract RIDs and embeddings
rids = df['RID']
embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
embeddings = df[embedding_cols].values

print(f"\n✓ Extracted data:")
print(f"  Number of seizures: {len(rids)}")
print(f"  Unique patients: {rids.nunique()}")
print(f"  Embedding shape: {embeddings.shape}")

# Show first few rows
print(f"\n✓ First 3 rows (metadata only):")
print(df[['RID', 'IEEGname', 'start', 'end']].head(3))

# Load NumPy array
embeddings_np = np.load('/users/zcxu/ieeg_sz_embedding/data/seizure_embeddings_with_rid.npy')
print(f"\n✓ Loaded NumPy array:")
print(f"  Shape: {embeddings_np.shape}")
print(f"  Dtype: {embeddings_np.dtype}")

# Verify they match
assert np.allclose(embeddings, embeddings_np), "Embeddings don't match!"
print(f"\n✓ CSV and NumPy embeddings match!")

# Load metadata
metadata = pd.read_csv('/users/zcxu/ieeg_sz_embedding/data/seizure_embeddings_with_rid_metadata.csv')
print(f"\n✓ Loaded metadata:")
print(f"  Shape: {metadata.shape}")
print(f"  Columns: {list(metadata.columns)}")

# Show patient statistics
print(f"\n✓ Top 10 patients by seizure count:")
print(rids.value_counts().head(10))

# Example: Compute similarity for one patient
from sklearn.metrics.pairwise import cosine_similarity

example_patient = rids.value_counts().index[0]  # Patient with most seizures
patient_embeddings = embeddings[rids == example_patient]

if len(patient_embeddings) > 1:
    similarity = cosine_similarity(patient_embeddings)
    avg_similarity = similarity[np.triu_indices_from(similarity, k=1)].mean()
    print(f"\n✓ Example: Patient {example_patient}")
    print(f"  Number of seizures: {len(patient_embeddings)}")
    print(f"  Average pairwise similarity: {avg_similarity:.3f}")

print(f"\n" + "="*60)
print("✅ All files exported successfully and verified!")
print("="*60)
