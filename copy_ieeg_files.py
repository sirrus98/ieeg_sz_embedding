"""
Script to copy iEEG files from network location to local drive.
Copies all files under 'ieeg' folders while maintaining directory structure.
Skips 'anat' folders and 'derivatives' folders (MRI data).
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
SOURCE_ROOT = r"\\sauce.seas.upenn.edu\data\Human_Data\CNT_iEEG_BIDS"
DEST_ROOT = r"C:\Users\sirrus\Desktop\ieeg_sz_embedding\CNT_iEEG_BIDS_local"

# Folders to skip entirely
SKIP_FOLDERS = {'derivatives', 'anat'}

# Root files to copy (BIDS metadata)
ROOT_FILES = [
    'dataset_description.json',
    'participants.json',
    'participants.tsv',
    'README'
]


def should_skip_path(path_parts):
    """Check if any part of the path should be skipped."""
    return any(skip in path_parts for skip in SKIP_FOLDERS)


def parse_tree_file(tree_file):
    """
    Parse the tree.txt file to extract directory structure.
    Returns list of relative paths for files under 'ieeg' folders.
    """
    with open(tree_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    paths_to_copy = []
    path_stack = []  # Stack to track current path at each depth
    skip_until_depth = None  # Depth to skip until (for skipping entire folders)
    in_ieeg_folder = False
    ieeg_base_depth = None
    
    for line_num, line in enumerate(lines, 1):
        # Remove line number if present (format: "   123|")
        if '|' in line[:10]:
            line = line[line.find('|')+1:]
        
        # Remove trailing newline
        line = line.rstrip('\n\r')
        
        # Skip empty lines or root marker
        if not line or line.strip() == '.' or not line.strip():
            continue
        
        # Calculate depth by counting tree levels
        # Tree uses: "│\xa0\xa0 " (│ + 2 non-breaking spaces + 1 space) for each level
        # Items end with "├── " or "└── "
        # \xa0 is non-breaking space (U+00A0)
        depth = 0
        temp_line = line
        
        # Count how many "│\xa0\xa0 " patterns we have
        while temp_line.startswith('│\xa0\xa0 '):
            depth += 1
            temp_line = temp_line[4:]  # Remove "│\xa0\xa0 "
        
        # Now temp_line should start with ├──  or └── 
        # Extract item name
        if temp_line.startswith('├── ') or temp_line.startswith('└── '):
            item_name = temp_line[4:].strip()
        elif temp_line.startswith('├─') or temp_line.startswith('└─'):
            item_name = temp_line[2:].strip()
        else:
            # Fallback: strip all tree characters
            item_name = temp_line.lstrip('│├└─ \xa0').strip()
        
        if not item_name:
            continue
        
        # Adjust path stack based on depth
        path_stack = path_stack[:depth] + [item_name]
        
        # Check if we should skip this path (derivatives, anat folders)
        if skip_until_depth is not None:
            if depth <= skip_until_depth:
                skip_until_depth = None
                in_ieeg_folder = False
                ieeg_base_depth = None
            else:
                continue
        
        # Check if current item is a folder we want to skip
        if item_name in SKIP_FOLDERS:
            skip_until_depth = depth
            continue
        
        # Check if we just entered an ieeg folder
        if item_name == 'ieeg':
            in_ieeg_folder = True
            ieeg_base_depth = depth
            continue
        
        # Check if we left the ieeg folder
        if in_ieeg_folder and depth <= ieeg_base_depth:
            in_ieeg_folder = False
            ieeg_base_depth = None
        
        # Add files to copy list
        is_file = '.' in item_name  # Files have extensions
        
        # Copy if: in ieeg folder OR root-level BIDS file
        if in_ieeg_folder and is_file:
            rel_path = '/'.join(path_stack)
            paths_to_copy.append(rel_path)
        elif depth == 0 and item_name in ROOT_FILES:
            paths_to_copy.append(item_name)
    
    return paths_to_copy


def copy_files(paths_to_copy, source_root, dest_root, dry_run=True):
    """
    Copy files from source to destination.
    
    Args:
        paths_to_copy: List of relative paths to copy
        source_root: Source root directory
        dest_root: Destination root directory
        dry_run: If True, only print what would be copied without actually copying
    """
    print(f"{'DRY RUN - ' if dry_run else ''}Copying files from:")
    print(f"  Source: {source_root}")
    print(f"  Destination: {dest_root}")
    print(f"\nTotal files to copy: {len(paths_to_copy)}")
    print("-" * 80)
    
    if not dry_run:
        os.makedirs(dest_root, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    for rel_path in tqdm(paths_to_copy, desc="Copying files"):
        # Convert forward slashes to backslashes for Windows
        rel_path_win = rel_path.replace('/', '\\')
        
        source_path = os.path.join(source_root, rel_path_win)
        dest_path = os.path.join(dest_root, rel_path_win)
        
        try:
            if dry_run:
                # In dry run, just check if source exists
                if os.path.exists(source_path):
                    copied_count += 1
                else:
                    print(f"  WARNING: Source not found: {source_path}")
                    skipped_count += 1
            else:
                # Create destination directory if it doesn't exist
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                
                # Copy the file
                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                else:
                    print(f"  WARNING: Source not found: {source_path}")
                    skipped_count += 1
                    
        except Exception as e:
            print(f"  ERROR copying {rel_path}: {e}")
            error_count += 1
    
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  {'Would be copied' if dry_run else 'Copied'}: {copied_count} files")
    if skipped_count > 0:
        print(f"  Skipped (not found): {skipped_count} files")
    if error_count > 0:
        print(f"  Errors: {error_count} files")
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Copy iEEG files from network location to local drive'
    )
    parser.add_argument(
        '--tree-file',
        default='cnt_ieeg_tree.txt',
        help='Path to the tree file (default: cnt_ieeg_tree.txt)'
    )
    parser.add_argument(
        '--source',
        default=SOURCE_ROOT,
        help=f'Source directory (default: {SOURCE_ROOT})'
    )
    parser.add_argument(
        '--dest',
        default=DEST_ROOT,
        help=f'Destination directory (default: {DEST_ROOT})'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually copy files (default is dry-run)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.tree_file):
        print(f"ERROR: Tree file not found: {args.tree_file}")
        return 1
    
    print("Parsing tree file...")
    paths_to_copy = parse_tree_file(args.tree_file)
    
    if not paths_to_copy:
        print("ERROR: No files found to copy!")
        return 1
    
    # Perform dry run or actual copy
    copy_files(paths_to_copy, args.source, args.dest, dry_run=not args.execute)
    
    if not args.execute:
        print("\nThis was a DRY RUN. Use --execute to actually copy files.")
    
    return 0


if __name__ == '__main__':
    exit(main())
