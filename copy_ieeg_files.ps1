# PowerShell script to copy iEEG files from network location to local drive
# Uses robocopy for efficient network file transfers

# Configuration
$SOURCE_ROOT = "\\sauce.seas.upenn.edu\data\Human_Data\CNT_iEEG_BIDS"
$DEST_ROOT = "C:\Users\sirrus\Desktop\ieeg_sz_embedding\CNT_iEEG_BIDS_local"

# Create destination directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $DEST_ROOT | Out-Null

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "Copying iEEG files from network location to local drive" -ForegroundColor Green
Write-Host "Source: $SOURCE_ROOT" -ForegroundColor Yellow
Write-Host "Destination: $DEST_ROOT" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

# First, copy root-level BIDS metadata files
Write-Host "Copying root-level BIDS metadata files..." -ForegroundColor Green
$rootFiles = @(
    "dataset_description.json",
    "participants.json",
    "participants.tsv",
    "README"
)

foreach ($file in $rootFiles) {
    $sourcePath = Join-Path $SOURCE_ROOT $file
    $destPath = Join-Path $DEST_ROOT $file
    if (Test-Path $sourcePath) {
        Copy-Item -Path $sourcePath -Destination $destPath -Force
        Write-Host "  Copied: $file" -ForegroundColor Gray
    }
}
Write-Host ""

# Get all subject folders
Write-Host "Finding subject folders..." -ForegroundColor Green
$subjects = Get-ChildItem -Path $SOURCE_ROOT -Directory -Filter "sub-*" | Select-Object -ExpandProperty Name
Write-Host "Found $($subjects.Count) subjects" -ForegroundColor Gray
Write-Host ""

# For each subject, copy ieeg folders (skip derivatives and anat)
$totalSubjects = $subjects.Count
$currentSubject = 0

foreach ($subject in $subjects) {
    $currentSubject++
    Write-Host "[$currentSubject/$totalSubjects] Processing $subject..." -ForegroundColor Green
    
    $subjectSource = Join-Path $SOURCE_ROOT $subject
    $subjectDest = Join-Path $DEST_ROOT $subject
    
    # Get session folders
    $sessions = Get-ChildItem -Path $subjectSource -Directory -Filter "ses-*" -ErrorAction SilentlyContinue
    
    if ($sessions) {
        foreach ($session in $sessions) {
            $sessionName = $session.Name
            $ieegSource = Join-Path $session.FullName "ieeg"
            $ieegDest = Join-Path $subjectDest $sessionName
            
            # Check if ieeg folder exists
            if (Test-Path $ieegSource) {
                # Create destination session folder
                New-Item -ItemType Directory -Force -Path $ieegDest | Out-Null
                
                # Use robocopy to copy ieeg folder
                # /E = copy subdirectories including empty ones
                # /NFL = no file list (less verbose)
                # /NDL = no directory list
                # /NJH = no job header
                # /NJS = no job summary
                # /NC = no class (file size/time)
                # /NS = no size
                # /NP = no progress
                # /MT:8 = multi-threaded (8 threads)
                # /R:3 = retry 3 times
                # /W:5 = wait 5 seconds between retries
                
                Write-Host "  Copying $sessionName\ieeg..." -ForegroundColor Gray
                $robocopyArgs = @(
                    $ieegSource,
                    (Join-Path $ieegDest "ieeg"),
                    "/E",
                    "/MT:8",
                    "/R:3",
                    "/W:5",
                    "/NFL",
                    "/NDL",
                    "/NJH",
                    "/NJS"
                )
                
                $result = robocopy @robocopyArgs
                
                # Robocopy exit codes: 0-7 are success, 8+ are errors
                $exitCode = $LASTEXITCODE
                if ($exitCode -ge 8) {
                    Write-Host "    WARNING: robocopy returned exit code $exitCode" -ForegroundColor Yellow
                }
            }
        }
    } else {
        # No session folders, check if there's a direct ieeg folder
        $ieegSource = Join-Path $subjectSource "ieeg"
        if (Test-Path $ieegSource) {
            New-Item -ItemType Directory -Force -Path $subjectDest | Out-Null
            Write-Host "  Copying ieeg..." -ForegroundColor Gray
            
            $robocopyArgs = @(
                $ieegSource,
                (Join-Path $subjectDest "ieeg"),
                "/E",
                "/MT:8",
                "/R:3",
                "/W:5",
                "/NFL",
                "/NDL",
                "/NJH",
                "/NJS"
            )
            
            $result = robocopy @robocopyArgs
            
            $exitCode = $LASTEXITCODE
            if ($exitCode -ge 8) {
                Write-Host "    WARNING: robocopy returned exit code $exitCode" -ForegroundColor Yellow
            }
        }
    }
}

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "Copy complete!" -ForegroundColor Green
Write-Host "Destination: $DEST_ROOT" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
