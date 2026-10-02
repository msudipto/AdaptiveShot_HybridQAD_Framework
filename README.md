# Adaptive-Shot Hybrid QAD Framework

Reliability-Aware Finite-Shot Hybrid Quantum Anomaly Detection for Tactile Internet Security

---

## Abstract

This repository presents the full experimental and implementation workflow for Adaptive-Shot Variational Quantum Circuit (AS-VQC) inference, a validation-calibrated measurement-allocation method for a hybrid quantum neural network used in Tactile-Internet-oriented anomaly detection.

The framework combines leakage-safe CESNET-TimeSeries24-derived benchmark construction, analytic hybrid-QNN training, fixed-shot and adaptive-shot finite-measurement evaluation, matched-budget controls, hierarchical statistical aggregation, and publication-ready figure generation.

The implementation is organized for research-grade reproducibility of the accepted NeurIPS 2026 SaTQuML Workshop paper, including Random, Entity-Group, and Temporal holdouts; five trained checkpoints per protocol; 128/256/512/1024-shot evaluation; and AS-VQC-90/95/99 calibration policies.

---

## Repository Structure

```text
AdaptiveShot-HybridQAD-Framework/
├── .github/
│   └── workflows/                       # Lightweight CI / syntax validation
├── artifacts/                           # Curated summaries, figures, and manuscript-facing outputs
├── config/                              # Experiment configuration
├── data/                                # Dataset placement instructions and local data layout
├── src/                                 # Core preprocessing, training, evaluation, aggregation, and plotting code
│
├── .gitattributes                       # Text/LFS tracking rules
├── .gitignore                           # Local data, environments, caches, and large-run exclusions
├── LICENSE                              # MIT License
├── README.md                            # Repository overview and reproducibility guide
├── commands.txt                         # Command reference
├── requirements.txt                     # Pinned Python dependency specification
├── run_pipeline.ps1                     # End-to-end Windows PowerShell execution pipeline
└── run_adaptive-shot_hybrid-qad.ps1     # Paper-specific alias for the same full pipeline
```

---

## Environment Configuration

### Requirements

- Python 3.12 recommended
- PyTorch
- PennyLane
- PennyLane-Lightning
- NumPy
- SciPy
- Pandas
- Scikit-learn
- Matplotlib
- PyYAML
- Windows PowerShell for the provided pipeline scripts
- Git LFS only when intentionally versioning large research assets

### Installation

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Repository Setup

Clone the repository and initialize the local environment:

```powershell
git clone <repository-url>
cd AdaptiveShot-HybridQAD-Framework
```

If intentionally versioned LFS assets are present:

```powershell
git lfs install
git lfs pull
```

---

## Experimental Methodology

### Data Preparation

The workflow uses CESNET-TimeSeries24 10-minute IP-address aggregate telemetry as an operational proxy benchmark. The raw dataset is not redistributed. A two-pass uniform sample constructs the 4,875-record candidate pool before fitted preprocessing or pseudo-label generation.

### Preprocessing Workflow

The preprocessing and split stages implement:

- train-only robust median/IQR scaling
- train-derived IQR filtering
- train-derived statistical-anomaly pseudo-labels
- Random, Entity-Group, and Temporal holdouts
- seeds 42--46 for repeated model evaluation
- saved split manifests for reproducibility

All data placement conventions are documented in `data/README.md`, and experiment settings are version-controlled in `config/adaptive_shots.yaml`.

### Hybrid QNN Training

The model pipeline implements a 12-feature classical embedder, a 12-qubit/two-layer variational quantum circuit, and a classical prediction head. Analytic expectation-value training selects checkpoints by validation ROC-AUC, and the deployment threshold is selected from validation data using Youden's statistic.

### Adaptive-Shot Evaluation

Finite-shot evaluation includes:

- Fixed-128, Fixed-256, Fixed-512, and Fixed-1024 baselines
- AS-VQC-90, AS-VQC-95, and AS-VQC-99
- matched-budget shuffled allocation control
- cumulative 128 → 256 → 512 → 1024 shot escalation
- ten test-time measurement realizations per non-analytic policy
- paired seed-level bootstrap comparisons

### Execution

For the default Windows workflow, run:

```powershell
.\run_pipeline.ps1
```

The paper-specific command below executes the same complete pipeline:

```powershell
.\run_adaptive-shot_hybrid-qad.ps1
```

The pipeline coordinates candidate preparation, leakage-safe split construction, 15 QNN training runs, fixed/adaptive-shot evaluation, hierarchical aggregation, and generation of four two-panel manuscript figures.

---

## Result Summary

| Component | Purpose | Research Role |
| --- | --- | --- |
| Candidate + Split Pipeline | Builds the 4,875-record leakage-safe benchmark | Prevents fitted preprocessing and pseudo-label leakage |
| Hybrid QNN | Produces analytic anomaly probabilities | Provides the trained quantum-classical scorer evaluated under finite shots |
| Adaptive-Shot Controller | Escalates measurements near the fixed threshold | Tests reliability-aware measurement allocation |
| Matched-Budget Control | Randomly reallocates the same realized budgets | Separates targeted allocation from measurement quantity |
| Hierarchical Aggregation | Averages repetitions within checkpoint, then summarizes seeds | Preserves the intended statistical unit of analysis |
| Figure Workflow | Produces manuscript-facing diagnostics and composites | Supports transparent reporting and camera-ready reproduction |

---

## Reproducibility Notes

- All experiment behavior is governed through `config/adaptive_shots.yaml`.
- The candidate sample and Random/Group training seeds are fixed for repeatability.
- Temporal evaluation uses a fixed chronological split while training randomness varies across seeds.
- Measurement repetitions are averaged within each trained checkpoint before across-seed summaries.
- Large checkpoints, raw data, repeated realizations, caches, and logs remain untracked unless intentionally archived.
- The curated `artifacts/` directory contains lightweight summaries and publication outputs that can be compared against a fresh reproduction.
- Dataset version, preprocessing assumptions, split logic, pseudo-label construction, and finite-shot settings should remain documented for manuscript-facing experiments.

---

## Data Notes

The repository does not redistribute the raw CESNET-TimeSeries24 dataset. See `data/README.md` for the expected local directory structure, acquisition/placement guidance, processed-output locations, and data-sharing restrictions.

---

## Artifact Policy

The `artifacts/` directory contains curated, lightweight research outputs such as:

- summary statistics used in the manuscript
- paired-bootstrap comparisons
- final diagnostic figures
- four two-panel manuscript figures
- aggregation and figure manifests

Large model checkpoints, per-record repeated finite-shot outputs, temporary runs, and caches are regenerable and are excluded from the public repository by default.

---

## Figures and Publication Assets

Publication assets are generated in stable formats such as PDF and PNG. The four composite manuscript figures can be regenerated from the saved summary artifacts with:

```powershell
python -m src.make_combined_figures
```

This plotting step does not retrain the QNN or rerun finite-shot inference.

---

## Authors and Collaborators

**Mubassir Serneabat Sudipto**  
Department of Electrical and Computer Engineering, College of Engineering, Iowa State University, Ames, Iowa, USA  
Email: msudipto@iastate.edu

**Shakil Ahmed**  
Department of Computer Science, College of Computing, Grand Valley State University, Allendale, Michigan, USA  
Email: ahmeshak@gvsu.edu

**Ashfaq Khokhar, Fellow, IEEE**  
Carl R. Ice College of Engineering, Kansas State University, Manhattan, Kansas, USA  
Email: akhokhar@k-state.edu

**Samir M. Iqbal**  
College of Computing, Grand Valley State University, Allendale, Michigan, USA  
Email: iqbalsa@gvsu.edu

---

## Citation

If you use this repository in academic work, please cite the associated paper and, where appropriate, the software repository itself.

```bibtex
@misc{adaptive_shot_hybrid_qad_2026,
  author = {Mubassir Serneabat Sudipto and Shakil Ahmed and Ashfaq Khokhar and Samir M. Iqbal},
  title  = {Adaptive-Shot Hybrid Quantum Anomaly Detection for Tactile Internet Security: Reliability-Aware Measurement Allocation Under Resource Constraints},
  year   = {2026},
  note   = {Accepted at the NeurIPS 2026 SaTQuML Workshop; accompanying reproducibility repository}
}
```

---

## License

This repository is released under the MIT License. See `LICENSE` for complete terms. CESNET-TimeSeries24 remains subject to its original dataset license and terms of use.

---

## Acknowledgment

This research repository supports reproducible work on trustworthy quantum machine learning, finite-shot hybrid quantum inference, quantum security, and Tactile-Internet-oriented anomaly detection.

---

Correspondence: Mubassir Serneabat Sudipto; Shakil Ahmed; Ashfaq Khokhar; Samir M. Iqbal  
Emails: msudipto@iastate.edu, ahmeshak@gvsu.edu, akhokhar@k-state.edu, iqbalsa@gvsu.edu
