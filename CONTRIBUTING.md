# Contributing to Adaptive-Shot Hybrid QAD Framework

Thank you for your interest in contributing to the Adaptive-Shot Hybrid QAD Framework. We welcome community enhancements, optimization tweaks, and bug fixes that advance research in resource-constrained quantum machine learning and zero-trust security.

## Development Rules & Environment

1. **Environment Requirements**:
   Ensure your local virtual environment mirrors the project dependencies specified in `requirements.txt`:
   * Python 3.12 (recommended)
   * PyTorch
   * PennyLane

2. **Testing Pipeline Health**:
   Before submitting any changes, verify that your modifications do not break the end-to-end adaptive shot evaluation workflows or artifact generation scripts:
   ```powershell
   .\run_adaptive-shot_hybrid-qad.ps1
   .\run_pipeline.ps1
   ```
   To test standalone figure rendering from existing artifacts, run:
   ```bash
   python -m src.make_combined_figures
   ```

## Calibration & Policy Integrity

If you introduce adjustments to the shot-allocation mechanism or uncertainty thresholds, evaluate your changes against the baseline policies established for this framework:
* **Analytic Reference**: Exact-expectation baseline.
* **Fixed Budgets**: Uniform shot configurations (Fixed-128, Fixed-256, Fixed-512, Fixed-1024).
* **AS-VQC Policies**: Calibration methods using validation-error percentiles (AS-VQC-90, AS-VQC-95, AS-VQC-99).
* **Budget-Shuffled-AS95 Control**: Verifies targeted measurement allocation efficiency by shuffling budgets randomly across test records.

## Commit and Artifact Policies

* **Do Not Commit Raw Training Weights or Datasets**: Heavy checkpoint binaries, individual per-record simulation loops, or large tabular segments must remain untracked by Git.
* **Stable Figure Exports**: Modified evaluation charts, cumulative distribution functions, or data tables saved to the `artifacts/` folder should be generated in stable vector or raster layouts (`.png`, `.svg`, `.pdf`).
