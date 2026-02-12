"""
Configuration for benchmark classification tasks.

This file contains paths and hyperparameters for both:
1. Semiology classification (multi-class)
2. Ictal vs Interictal detection (binary)

IMPORTANT: Update the paths below to match your environment.
"""

import os
from pathlib import Path

class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    
    # ===== PATHS - UPDATE THESE FOR YOUR ENVIRONMENT =====
    # Data directory containing preprocessed seizure data (READ-ONLY - from pattnaik)
    DATA_DIR = "/mnt/leif/littlab/users/pattnaik/ieeg_sz_embedding/data"
    
    # Output directory for saving generated data (WRITE - to zcxu)
    OUTPUT_DATA_DIR = "/mnt/leif/littlab/users/zcxu/ieeg_sz_embedding/data"
    
    # Raw BIDS data directory for extracting interictal windows
    RAW_DATA_DIR = "/mnt/leif/littlab/data/Human_Data/CNT_iEEG_BIDS"
    
    # Checkpoint directory for pretrained models
    PRETRAINED_CKPT_DIR = "/mnt/leif/littlab/users/pattnaik/ieeg_sz_embedding/checkpoints"
    
    # Checkpoint directory for saving benchmark models
    BENCHMARK_CKPT_DIR = "./benchmark_checkpoints"
    
    # Logs directory
    LOGS_DIR = "./benchmark_logs"
    
    # Output directory for embeddings
    EMBEDDINGS_DIR = "./benchmark_embeddings"
    
    # ===== DATA FILES =====
    # Ictal (seizure) data - 10 second windows (READ from pattnaik)
    ICTAL_DATA_FILE = os.path.join(DATA_DIR, "all_sz_coords_data_spatiotemporal_regs_10s.pkl")
    
    # Interictal (non-seizure) data - will be created by extract_interictal_data.py (WRITE to zcxu)
    INTERICTAL_DATA_FILE = os.path.join(OUTPUT_DATA_DIR, "all_interictal_coords_data_spatiotemporal_regs_10s.pkl")
    
    # Metadata files
    SEIZURE_METADATA_FILE = os.path.join(DATA_DIR, "metadata/Manual validation.xlsx")
    RID_HUP_TABLE_FILE = os.path.join(DATA_DIR, "metadata/rid_hup_table.csv")
    ANNOTATIONS_FILE = os.path.join(DATA_DIR, "metadata/master_bipolars.csv")
    DKT_MAPPING_FILE = os.path.join(DATA_DIR, "atlases/luts/dktg_reordered.csv")
    
    # ===== MODEL CONFIGURATION =====
    # Pretrained model checkpoint (set to None to train from scratch)
    PRETRAINED_CHECKPOINT = None  # e.g., "checkpoints_10s/model_epoch-100-train_loss-0.12345.ckpt"
    
    # Whether to freeze encoder during fine-tuning
    FREEZE_ENCODER = False  # Set to True for feature extraction mode
    
    # ===== TRAINING HYPERPARAMETERS =====
    # General
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    RANDOM_SEED = 42
    
    # Optimization
    LEARNING_RATE = 1e-4  # Lower than self-supervised (1e-2) for fine-tuning
    WEIGHT_DECAY = 1e-5
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    
    # Learning rate scheduler
    LR_SCHEDULER = "ReduceLROnPlateau"
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    
    # Data splits (patient-stratified)
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # ===== SEMIOLOGY CLASSIFICATION SPECIFIC =====
    # Minimum samples per semiology class (filter out rare classes)
    MIN_SAMPLES_PER_CLASS = 10
    
    # Use class weights to handle imbalance
    USE_CLASS_WEIGHTS = True
    
    # ===== ICTAL/INTERICTAL DETECTION SPECIFIC =====
    # Number of interictal windows to extract per patient
    INTERICTAL_WINDOWS_PER_PATIENT = 2
    
    # Buffer time (seconds) to avoid peri-ictal contamination
    INTERICTAL_BUFFER_TIME = 3600  # 1 hour
    
    # Duration of each interictal window (seconds)
    INTERICTAL_WINDOW_DURATION = 10  # Match ictal window duration
    
    # ===== MODEL ARCHITECTURE =====
    # From model_config.py
    CHANNEL_BUFFER_SIZE = 150
    TARGET_SAMPLING_RATE = 256  # Hz
    MAX_REGIONS = 41
    
    # Expected input shape
    INPUT_TIMESTEPS = TARGET_SAMPLING_RATE * INTERICTAL_WINDOW_DURATION  # 2560
    
    # ===== LOGGING =====
    LOG_EVERY_N_STEPS = 10
    
    # ===== DEVICE =====
    ACCELERATOR = "gpu"  # or "cpu"
    DEVICES = 1
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.OUTPUT_DATA_DIR, exist_ok=True)
        os.makedirs(cls.BENCHMARK_CKPT_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        os.makedirs(cls.EMBEDDINGS_DIR, exist_ok=True)
        print(f"✓ Created directories:")
        print(f"  - Output data: {cls.OUTPUT_DATA_DIR}")
        print(f"  - Checkpoints: {cls.BENCHMARK_CKPT_DIR}")
        print(f"  - Logs: {cls.LOGS_DIR}")
        print(f"  - Embeddings: {cls.EMBEDDINGS_DIR}")


if __name__ == "__main__":
    # Test configuration
    config = BenchmarkConfig()
    config.create_directories()
    
    print("\n===== Benchmark Configuration =====")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Raw data directory: {config.RAW_DATA_DIR}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Max epochs: {config.MAX_EPOCHS}")
    print(f"Input shape: ({config.CHANNEL_BUFFER_SIZE}, {config.INPUT_TIMESTEPS})")
    print("\nNote: Update paths in config_benchmark.py to match your environment!")
