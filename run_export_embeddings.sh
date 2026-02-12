#!/bin/bash
# Script to export seizure embeddings with patient IDs (RID)
# This script handles environment setup and runs the appropriate export method

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Seizure Embedding Export"
echo "=========================================="
echo ""

# Check for conda environment (adjust if your environment has a different name)
if [ -z "$CONDA_PREFIX" ]; then
    echo "⚠ No conda environment detected."
    echo "Please activate your conda environment first, e.g.:"
    echo "  conda activate cis522"
    echo "  # or"
    echo "  conda activate your_env_name"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✓ Conda environment: $CONDA_PREFIX"
echo ""

# Check which method to use
echo "Select export method:"
echo "  1) Generate embeddings from scratch (requires model checkpoint)"
echo "  2) Export from existing notebook variables (run in notebook)"
echo "  3) Export from pre-saved embeddings (if available)"
echo ""

# Auto-detect if we're in a notebook environment
if [ -t 0 ]; then
    # Interactive terminal
    read -p "Enter choice [1-3]: " choice
else
    # Non-interactive, default to method 1
    choice=1
    echo "Running in non-interactive mode, using method 1 (generate from scratch)"
fi

echo ""

case $choice in
    1)
        echo "Method 1: Generating embeddings from scratch..."
        echo "This will load the model checkpoint and process all seizures."
        echo "This may take several minutes..."
        echo ""
        python code/generate_embeddings.py
        ;;
    2)
        echo "Method 2: Export from notebook"
        echo ""
        echo "To use this method:"
        echo "  1. Open code/test_nb.ipynb in Jupyter"
        echo "  2. Run cells up to Cell 17 (where sz_embeddings is created)"
        echo "  3. Run the following in a new cell:"
        echo ""
        cat code/export_from_notebook.py
        echo ""
        ;;
    3)
        echo "Method 3: Exporting from pre-saved embeddings..."
        python code/export_existing_embeddings.py
        ;;
    *)
        echo "Invalid choice. Please select 1, 2, or 3."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
