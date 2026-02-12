"""
Generate embeddings for seizure data and export with patient IDs (RID)
"""

import os
from os.path import join as ospj
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Import necessary modules
from config import CONFIG
from model_v5 import MultivarWav2Vec2
from model_wrapper import ModelV3WrapperWithQueue
from dataset import SzDatasetRegsFull
from model_config import MODEL_CONFIG


def load_model(ckpt_path, device='cpu'):
    """Load the trained model from checkpoint"""
    print(f"Loading model from: {ckpt_path}")
    model_nn = MultivarWav2Vec2(MODEL_CONFIG)
    # load_from_checkpoint is a classmethod, call it on the class not an instance
    wrapper = ModelV3WrapperWithQueue.load_from_checkpoint(
        ckpt_path, model=model_nn
    )
    model = wrapper.model
    model = model.eval()
    model = model.to(device)
    return model


def generate_embeddings(model, dataset, device='cpu', batch_size=1):
    """
    Generate embeddings for all samples in the dataset
    
    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        metadata: pandas DataFrame with patient IDs and other info
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_tokens = np.zeros((len(dataset), MODEL_CONFIG.prototype_dim))
    all_regs = []
    all_eegs = []
    all_ch_names = []
    
    print("Generating embeddings...")
    for batch in tqdm(dataloader, total=len(dataloader)):
        # Get the inputs
        signals, regs, ch_names, idx = batch
        regs = regs.long().to(device)
        signals = signals.float().to(device)
        signals = signals[:, :, :int(256*10)]  # Take first 10 seconds
        
        # Forward pass
        with torch.no_grad():
            output = model(signals, regs)
        
        all_tokens[idx] = output.detach().cpu().numpy()
        all_regs.append(regs.cpu().numpy())
        all_eegs.append(signals.cpu().numpy())
        all_ch_names.append([i[0] for i in ch_names])
    
    # Filter out samples with NaN values
    valid_mask = ~np.any(np.isnan(all_tokens), axis=-1)
    sz_embeddings = all_tokens[valid_mask]
    sz_metadata = dataset.metadata[valid_mask]
    
    print(f"Generated {len(sz_embeddings)} valid embeddings out of {len(dataset)} total samples")
    print(f"Embedding dimension: {sz_embeddings.shape[1]}")
    
    return sz_embeddings, sz_metadata


def export_embeddings(embeddings, metadata, output_path):
    """
    Export embeddings with patient IDs to CSV
    
    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        metadata: pandas DataFrame with patient information
        output_path: path to save the CSV file
    """
    # Create a dataframe with RID and embeddings
    export_df = pd.DataFrame()
    
    # Add patient ID (RID)
    export_df['RID'] = metadata['Patient'].values
    
    # Add other metadata if available
    if 'IEEGID' in metadata.columns:
        export_df['IEEGID'] = metadata['IEEGID'].values
    if 'IEEGname' in metadata.columns:
        export_df['IEEGname'] = metadata['IEEGname'].values
    if 'start' in metadata.columns:
        export_df['start'] = metadata['start'].values
    if 'end' in metadata.columns:
        export_df['end'] = metadata['end'].values
    
    # Add each embedding dimension as a column (more efficient way)
    embedding_cols = {f'embedding_{i}': embeddings[:, i] for i in range(embeddings.shape[1])}
    embedding_df = pd.DataFrame(embedding_cols)
    export_df = pd.concat([export_df, embedding_df], axis=1)
    
    # Save to CSV
    export_df.to_csv(output_path, index=False)
    print(f"Exported embeddings to: {output_path}")
    print(f"Shape: {export_df.shape}")
    print(f"Columns: RID + {embeddings.shape[1]} embedding dimensions")
    
    # Also save as numpy file for easier loading
    np_output_path = output_path.replace('.csv', '.npy')
    np.save(np_output_path, embeddings)
    print(f"Also saved embeddings as numpy array to: {np_output_path}")
    
    # Save metadata separately
    metadata_output_path = output_path.replace('.csv', '_metadata.csv')
    metadata.to_csv(metadata_output_path, index=False)
    print(f"Saved metadata to: {metadata_output_path}")


def main():
    """Main function to generate and export embeddings"""
    
    # Path to the best model checkpoint (from Cell 15-16 in the notebook)
    # This is the best model across different seeds (seed_3 had the lowest loss)
    # ckpt_path = "/mnt/leif/littlab/users/pattnaik/ieeg_sz_embedding/checkpoints/checkpoints_seed_3/model_epoch-epoch=499-train_loss=2.21505.ckpt"
    
    # Alternative checkpoints if needed:
    ckpt_path = "/mnt/leif/littlab/users/pattnaik/ieeg_sz_embedding/checkpoints/checkpoints_10s/model_epoch-epoch=499-train_loss=2.95853.ckpt"
    
    # Check if checkpoint exists
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found at {ckpt_path}")
        print("Please update the ckpt_path variable with the correct path")
        return
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(ckpt_path, device=device)
    
    # Load dataset
    print("Loading dataset...")
    dataset = SzDatasetRegsFull()
    print(f"Dataset size: {len(dataset)}")
    
    # Generate embeddings
    embeddings, metadata = generate_embeddings(model, dataset, device=device)
    
    # Export to file - use a directory the user has write permissions for
    # Try user's project directory first
    output_dir = "/users/zcxu/ieeg_sz_embedding/data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = ospj(output_dir, "seizure_embeddings_with_rid.csv")
    export_embeddings(embeddings, metadata, output_path)
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total embeddings generated: {len(embeddings)}")
    print(f"Number of unique patients: {metadata['Patient'].nunique()}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print("\nPatients with embeddings:")
    print(metadata['Patient'].value_counts())


if __name__ == "__main__":
    main()
