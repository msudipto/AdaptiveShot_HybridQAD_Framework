# run_pipeline.ps1
# End-to-end pipeline runner for the Adaptive-Shot Hybrid QAD experiments (Windows PowerShell)

$ErrorActionPreference = "Stop"

# Step 0: Move to repository root.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Write-Host "`n[pipeline] Repo root:" (Get-Location) -ForegroundColor Cyan

# Step 1: Activate virtual environment if present.
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "`n[pipeline] Activating virtual environment..." -ForegroundColor Green
    .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "`n[pipeline] WARNING: .venv not found." -ForegroundColor Yellow
    Write-Host "Create one with: py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
}

# Step 2: Prepare the Python environment.
if (Test-Path ".\requirements.txt") {
    Write-Host "`n[pipeline] Installing pinned dependencies..." -ForegroundColor Green
    python -m pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
} else {
    Write-Host "`n[pipeline] WARNING: requirements.txt not found; skipping dependency installation." -ForegroundColor Yellow
}

# Step 3: Candidate preparation.
Write-Host "`n[pipeline] [1/6] Preparing 4,875-record candidate pool..." -ForegroundColor Green
python -m src.prepare_candidate --config config/adaptive_shots.yaml

# Step 4: Leakage-safe splits.
Write-Host "`n[pipeline] [2/6] Building Random / Group / Temporal splits..." -ForegroundColor Green
python -m src.make_splits --config config/adaptive_shots.yaml

# Step 5: QNN training.
Write-Host "`n[pipeline] [3/6] Training 15 source QNN checkpoints..." -ForegroundColor Green
python -m src.train_qnn --config config/adaptive_shots.yaml --all

# Step 6: Fixed/adaptive-shot evaluation.
Write-Host "`n[pipeline] [4/6] Running fixed-shot and adaptive-shot evaluation..." -ForegroundColor Green
python -m src.adaptive_shots --config config/adaptive_shots.yaml --all

# Step 7: Hierarchical aggregation.
Write-Host "`n[pipeline] [5/6] Aggregating hierarchical results..." -ForegroundColor Green
python -m src.aggregate --config config/adaptive_shots.yaml

# Step 8: Publication figures.
Write-Host "`n[pipeline] [6/6] Generating publication-ready composite figures..." -ForegroundColor Green
python -m src.make_combined_figures

Write-Host "`n[pipeline] Pipeline completed." -ForegroundColor Cyan
Write-Host "[pipeline] See ./artifacts, ./data/processed, and ./data/processed/splits" -ForegroundColor Cyan
