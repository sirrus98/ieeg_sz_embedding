# iEEG Data Copy Instructions

This folder contains scripts to copy iEEG data from the network location to your local drive while skipping MRI/anatomical data.

## Network Source
- **Source:** `\\sauce.seas.upenn.edu\data\Human_Data\CNT_iEEG_BIDS`
- **Destination:** `C:\Users\sirrus\Desktop\ieeg_sz_embedding\CNT_iEEG_BIDS_local`

## What Gets Copied
- ✅ All files under `ieeg` folders
- ✅ Root-level BIDS metadata files (dataset_description.json, participants.tsv, etc.)
- ✅ CT scans (if in same session folders as ieeg)
- ❌ `anat` folders (MRI anatomical data)
- ❌ `derivatives` folders (freesurfer MRI processing)

## Option 1: PowerShell Script (Recommended for Windows)

**Fastest option using robocopy with multi-threading.**

### Usage:

```powershell
# Run the PowerShell script
.\copy_ieeg_files.ps1
```

**Features:**
- Multi-threaded copying (8 threads) for faster network transfers
- Automatic retry on network errors
- Progress indication
- Built-in Windows tool (no dependencies)

---

## Option 2: Python Script

**More flexible, with dry-run option to preview what will be copied.**

### Prerequisites:
```bash
pip install tqdm
```

### Usage:

**Step 1: Dry run (preview what will be copied)**
```bash
python copy_ieeg_files.py
```

**Step 2: Actually copy files**
```bash
python copy_ieeg_files.py --execute
```

**Custom paths:**
```bash
python copy_ieeg_files.py --execute --source "\\network\path" --dest "C:\local\path"
```

**Features:**
- Dry-run mode to preview
- Progress bar with tqdm
- Detailed logging
- Cross-platform (works on Windows, Mac, Linux)

---

## Customization

### Change Destination Path

**PowerShell:**
Edit line 5 in `copy_ieeg_files.ps1`:
```powershell
$DEST_ROOT = "D:\YourPath\CNT_iEEG_BIDS_local"  # Change this
```

**Python:**
Either edit line 13 in `copy_ieeg_files.py`:
```python
DEST_ROOT = r"D:\YourPath\CNT_iEEG_BIDS_local"  # Change this
```
Or use command-line argument:
```bash
python copy_ieeg_files.py --execute --dest "D:\YourPath\CNT_iEEG_BIDS_local"
```

### Include Additional Files

If you want to also copy CT scans or other modalities:

**PowerShell:** Edit the script to add additional folder names in the robocopy section

**Python:** Modify the `should_skip_path()` function to include/exclude folders

---

## Troubleshooting

### Network Access Issues
- Make sure you're connected to the UPenn network (VPN if remote)
- Verify you have read access to `\\sauce.seas.upenn.edu\data\Human_Data\CNT_iEEG_BIDS`
- Try accessing the network path in File Explorer first

### Disk Space
- Check available disk space before starting
- iEEG files can be large (each subject may be several GB)
- Use dry-run mode to estimate total size

### PowerShell Execution Policy
If you get an error about execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Python Dependencies
If tqdm is not installed:
```bash
pip install tqdm
```

---

## Estimated Copy Time

Depends on:
- Number of subjects
- Network speed
- File sizes

With ~100 subjects, expect:
- **PowerShell (robocopy):** 30-60 minutes (multi-threaded)
- **Python:** 60-120 minutes (single-threaded)

---

## Notes

- The directory structure will be preserved: `sub-RID####/ses-clinical##/ieeg/`
- Both scripts skip files that already exist (can be resumed if interrupted)
- Robocopy is generally faster for large network transfers
- Python script provides more detailed progress and logging
