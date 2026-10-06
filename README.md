---
license: mit
library_name: pytorch
pipeline_tag: tabular-classification
tags:
  - pytorch
  - pennylane
  - quantum-machine-learning
  - quantum-neural-network
  - variational-quantum-classifier
  - hybrid-quantum-classical
  - adaptive-shot
  - finite-shot-inference
  - measurement-allocation
  - anomaly-detection
  - network-anomaly-detection
  - cybersecurity
  - zero-trust
  - tactile-internet
  - network-security
  - encrypted-traffic
  - cesnet
  - arxiv:2610.05835
---

# Adaptive-Shot Hybrid QAD Framework
## Adaptive-Shot Hybrid Quantum Anomaly Detection for Tactile Internet Security: Reliability-Aware Measurement Allocation Under Resource Constraints

[![arXiv](https://img.shields.io/badge/arXiv-2610.05835-b31b1b.svg)](https://arxiv.org/abs/2610.05835)
[![Hugging Face Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow.svg)](https://huggingface.co/papers/2610.05835)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2610.05835-blue.svg)](https://doi.org/10.48550/arXiv.2610.05835)
[![Workshop](https://img.shields.io/badge/NeurIPS%202026-SaTQuML%20Workshop-blue.svg)](https://arxiv.org/abs/2610.05835)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Paper:** [Adaptive-Shot Hybrid Quantum Anomaly Detection for Tactile Internet Security: Reliability-Aware Measurement Allocation Under Resource Constraints](https://arxiv.org/abs/2610.05835)  
**Hugging Face Paper:** [huggingface.co/papers/2610.05835](https://huggingface.co/papers/2610.05835)  
**Code:** [github.com/msudipto/AdaptiveShot_HybridQAD_Framework](https://github.com/msudipto/AdaptiveShot_HybridQAD_Framework)  
**DOI listed by arXiv:** [10.48550/arXiv.2610.05835](https://doi.org/10.48550/arXiv.2610.05835)  
**Venue:** **NeurIPS 2026 — Secure and Trustworthy Quantum Machine Learning (SaTQuML) Workshop**  
**Status:** Accepted for a **Short Oral Presentation**, as reported in the arXiv comments  
**Preprint:** arXiv:2610.05835v1, submitted October 5, 2026  
**Primary arXiv Category:** Cryptography and Security (`cs.CR`)  
**Cross-lists:** Machine Learning (`cs.LG`), Networking and Internet Architecture (`cs.NI`)

---

## Model Description

**AS-VQC (Adaptive-Shot Variational Quantum Circuit)** is a validation-calibrated measurement-allocation policy for hybrid quantum–classical anomaly inference in Tactile-Internet-oriented security.

Instead of assigning every record the same measurement budget, AS-VQC starts with **128 shots** and acquires additional measurements only when the estimated anomaly score remains near a fixed security threshold:

```text
128 → 256 → 512 → 1024 cumulative shots
```

The framework separates:

1. **Off-path anomaly evidence**, generated from mirrored aggregate-flow telemetry using a trained hybrid quantum neural network (QNN); and
2. **On-path deterministic enforcement**, which uses cached policies rather than waiting for quantum-model inference on the current request.

The primary policy is **AS-VQC-95**, calibrated with the 95th percentile of validation finite-shot errors. The percentile is an empirical calibration setting, **not a formal 95% guarantee of test-decision correctness**.

The implementation uses **PyTorch** and **PennyLane**. Training uses noiseless analytic expectations; inference experiments simulate ideal finite-shot measurements. The study evaluates measurement demand and decision stability, not physical quantum-hardware performance.

### Model Sources

- **Paper:** https://arxiv.org/abs/2610.05835
- **Full author version:** https://arxiv.org/html/2610.05835v1
- **Hugging Face Paper Page:** https://huggingface.co/papers/2610.05835
- **Source Code:** https://github.com/msudipto/AdaptiveShot_HybridQAD_Framework
- **DOI:** https://doi.org/10.48550/arXiv.2610.05835

---

## Authors

Author order follows the linked arXiv paper.

### Mubassir Serneabat Sudipto
Electrical and Computer Engineering, College of Engineering  
Iowa State University  
Ames, Iowa, USA  
Email: [msudipto@iastate.edu](mailto:msudipto@iastate.edu)

### Shakil Ahmed
Computer Science, College of Computing  
Grand Valley State University  
Allendale, Michigan, USA  
Email: [ahmeshak@gvsu.edu](mailto:ahmeshak@gvsu.edu)

### Ashfaq Khokhar
Carl R. Ice College of Engineering  
Kansas State University  
Manhattan, Kansas, USA  
Email: [akhokhar@ksu.edu](mailto:akhokhar@ksu.edu)

### Samir M. Iqbal
College of Computing  
Grand Valley State University  
Allendale, Michigan, USA  
Email: [iqbalsa@gvsu.edu](mailto:iqbalsa@gvsu.edu)

Affiliations and contact addresses follow the [paper](https://arxiv.org/html/2610.05835v1) and its [linked repository](https://github.com/msudipto/AdaptiveShot_HybridQAD_Framework).

---

## Model Architecture

The evaluated hybrid scorer contains the following stages:

1. **Feature preprocessing**
   - 12 payload-independent aggregate-flow features.
   - Training-only robust scaling, filtering limits, and pseudo-label boundaries.

2. **Classical embedding**
   - `12 → 64 → 12` with GELU activation.
   - Bounded encoding angles: `z = π × tanh(embedder(x_scaled))`.
   - **1,612 trainable parameters**.

3. **Quantum feature encoding**
   - **12 qubits**, initialized to the all-zero state.
   - One single-qubit `RY` encoding gate per qubit.

4. **Variational Quantum Circuit**
   - **2 layers**, each with three-angle rotations on every qubit and a directed nearest-neighbor CNOT chain.
   - **72 registered trainable parameters**, 24 rotation gates, and 22 CNOTs.

5. **Quantum measurement**
   - Pauli-Z readout on qubits **0 and 1**.
   - For this specific circuit/readout, backward causal support spans encoded wires **0–2**, enabling exact three-wire reconstruction of the joint readout distribution.

6. **Classical classification head**
   - `2 → 32 → 2` with GELU activation and softmax output.
   - **162 trainable parameters**; **1,846 total** across the full model.

7. **Adaptive-shot inference**
   - Reuse the trained checkpoint and validation-selected threshold without retraining.
   - At stage `S`, stop when `abs(p_hat_S - threshold) > delta_S,beta`.
   - Otherwise acquire only the additional shots needed for the next cumulative stage; stop all remaining records at 1024.

The margin `delta_S,beta` is fitted from validation sampling errors at stages 128, 256, and 512. Test labels and analytic test probabilities do not determine the stopping rule. Analytic test predictions serve as the evaluation reference.

Method source: [paper, Sections 3.3–3.6](https://arxiv.org/html/2610.05835v1#S3).

---

## Input

The model operates on **tabular aggregate-flow data**, not natural language, images, or quantum-origin traffic. The retained features are:

```text
n_flows                 n_packets
n_bytes                 n_dest_asn
n_dest_ports            n_dest_ip
tcp_udp_ratio_packets   tcp_udp_ratio_bytes
dir_ratio_packets       dir_ratio_bytes
avg_duration            avg_ttl
```

Source time-bin identifiers and entity keys support chronological and entity-disjoint partitioning; they are distinct from the 12 modeled features.

---

## Output

The scorer produces continuous statistical-anomaly evidence:

```text
0 <= anomaly_score <= 1

0 = normative record
1 = statistical-anomaly record
```

Finite-shot inference also records the final cumulative measurement budget:

```text
shots_used ∈ {128, 256, 512, 1024}
```

A thresholded decision uses the checkpoint's fixed, validation-selected operating threshold. **Neither the score nor disagreement with analytic inference is a verified probability of maliciousness or an attack-success measure.**

---

## Dataset and Experimental Data

The study uses **4,875 CESNET-TimeSeries24-derived records** from 10-minute IP-address aggregates. A seed-42 uniform sample without replacement is selected before fitted preprocessing from **6,863,540 eligible rows** in the active source view.

Training data alone determine medians, interquartile ranges (IQRs), `1.5 × IQR` filtering limits, and robust-scaled feature-norm quantiles. The 0.85 and 0.95 training quantiles identify suspicious and high-anomaly states; binary evaluation marks records at or above the 0.85 boundary as positive.

The candidate pool is not the final size of every partition: training-derived filtering changes the retained counts. Raw CESNET data are not redistributed; follow [data/README.md](data/README.md) and the dataset provider's terms.

### Important Data Qualification

These are **statistical pseudo-labels, not independently verified attack annotations**. CESNET telemetry is an operational proxy benchmark, not a dedicated TI or haptic-traffic capture. Reported discrimination and false-positive rates concern the constructed statistical benchmark.

Data and split source: [paper, Section 3 and Appendix A](https://arxiv.org/html/2610.05835v1#A1).

---

## Evaluation Protocols

The study constructs nominal **60/20/20 train/validation/test partitions** before applying the training-derived filter. Validation selects checkpoints, operating thresholds, and adaptive uncertainty margins; test data are reserved for reporting.

### Random Holdout

- Seeded record permutation, not the older two-part train/evaluation design.
- Seeds **42–46** vary splitting and training.
- Retained train/validation/test ranges: **1,746–1,801 / 591–602 / 576–624** records.

### Entity-Group Holdout

- Each entity key belongs to exactly one partition.
- Seeds **42–46** vary splitting and training.
- Retained train/validation/test ranges: **1,688–1,872 / 469–636 / 506–669** records.

### Temporal Holdout

- Ordered time bins separate training, validation, and test chronologically.
- Retained counts: **1,755 / 561 / 605** records.
- The chronological split is fixed; seeds **42–46** vary model/training randomness only.

These protocols separate record-wise evaluation, unseen-entity conditions, and later-time evaluation. Their variability is not interchangeable.

---

## Training Procedure

| Parameter | Value |
|---|---:|
| Input features / qubits | 12 / 12 |
| Variational layers | 2 |
| Total trainable parameters | 1,846 |
| Epochs | 20 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 2 × 10^-3 |
| Weight decay | 10^-4 |
| Gradient clipping | 1.0 |
| Seeds per holdout | 42–46 |
| Trained checkpoints | 5 per holdout; 15 total |
| Checkpoint selection | Maximum validation ROC-AUC |
| Threshold selection | Validation Youden's J |
| Training backend | Noiseless analytic expectations on CPU |

The embedder, VQC, and head are jointly optimized using inverse-frequency class-weighted negative log-likelihood, with weights normalized to unit mean. The first checkpoint is retained in an exact validation-AUC tie.

### Quantum Execution Setting

After analytic training, fixed-shot and adaptive-shot inference reuse the saved checkpoints **without retraining or threshold recalibration**. Ideal joint multinomial measurements preserve the relationship between the two measured outputs.

The paper reports **1,200 finite-shot policy realizations**: 3 holdouts × 5 checkpoints × 8 policies × 10 realizations. Including 15 analytic references gives **1,215 realization-level rows**. Measurement realizations are averaged within checkpoint before summarizing mean ± sample SD across five seeds. Paired intervals use 10,000 bootstrap resamples of seed-level differences, not independently trained-model counts inferred from repeated measurements.

---

## Evaluation Results

Results below are transcribed from the [paper's main results and supplementary diagnostics](https://arxiv.org/html/2610.05835v1#S4), not regenerated by this README update.

### Full-Hybrid QNN Results

**Primary AS-VQC-95 policy:** mean ± sample SD across five checkpoints after averaging measurement realizations within each checkpoint.

| Holdout | ROC-AUC | Average Precision | FPR | ECE |
|---|---:|---:|---:|---:|
| Random | **0.9956 ± 0.0014** | 0.9760 ± 0.0050 | 0.0299 ± 0.0042 | 0.0257 ± 0.0062 |
| Entity-Group | **0.9968 ± 0.0017** | 0.9835 ± 0.0092 | 0.0346 ± 0.0312 | 0.0174 ± 0.0038 |
| Temporal | **0.9980 ± 0.0010** | 0.9895 ± 0.0060 | 0.0252 ± 0.0104 | 0.0211 ± 0.0076 |

ECE denotes expected calibration error using 15 bins. FPR and ECE are fractions, not percentages, in this table.

### Measurement Savings and Decision Reliability

| Holdout | Average Shots ± SD | Saving vs. Fixed-1024 ± SD | Mean Decision Disagreement |
|---|---:|---:|---:|
| Random | **129.2 ± 0.4** | **87.39% ± 0.04 pp** | **0.771%** |
| Entity-Group | **276.9 ± 327.3** | **72.96% ± 31.97 pp** | **0.409%** |
| Temporal | **131.2 ± 2.0** | **87.19% ± 0.20 pp** | **0.635%** |

Here, `pp` means percentage points. Savings use `100 × (1 − average_shots / 1024)`; disagreement compares the finite-shot decision with the **analytic reference decision at the same threshold**, not with verified cyberattack ground truth. The displayed precision follows the paper; independently rounded columns may not reproduce exactly from one another.

The large Group SD is retained. In particular, Group seed 46 averages **862.4 shots per record**, unlike the other four Group checkpoints, which remain near the 128-shot floor. This is reported checkpoint-dependent demand, not a removed outlier.

### Comparison with Fixed-Shot and Matched-Budget Controls

AS-VQC-95 has lower decision disagreement than **Fixed-128** and **Budget-Shuffled-AS95** in all three holdouts. The latter preserves its realized shot-budget distribution but randomly assigns those budgets to records, testing targeted allocation rather than measurement quantity alone.

**Fixed-1024 remains more decision-stable than AS-VQC-95.** The savings therefore represent a reliability–measurement tradeoff, not equivalent reliability at lower cost.

The more conservative AS-VQC-99 policy gives the following observed means:

| Holdout | AS-VQC-99 Average Shots | AS-VQC-99 Disagreement | Fixed-512 Disagreement |
|---|---:|---:|---:|
| Random | 162.6 | 0.407% | 0.462% |
| Entity-Group | 432.9 | 0.248% | 0.333% |
| Temporal | 207.7 | 0.423% | 0.532% |

This comparison is descriptive; AS-VQC-95 remains the prespecified primary policy. None of these results establishes quantum computational advantage or physical-device energy or latency savings.

---

## Ablation Study

The controls in this paper isolate **measurement allocation**, not removal of the classical embedder, head, or VQC. All policies reuse the same selected QNN checkpoints.

| Policy | Controlled Change |
|---|---|
| **Analytic** | Exact-expectation reference |
| **Fixed-128 / 256 / 512 / 1024** | Uniform measurement budget per record |
| **AS-VQC-90** | Adaptive inference with 90th-percentile validation-error margins |
| **AS-VQC-95** | Primary adaptive inference with 95th-percentile margins |
| **AS-VQC-99** | More conservative 99th-percentile margins |
| **Budget-Shuffled-AS95** | Same realized AS-VQC-95 budgets, reassigned randomly across records |

The matched-budget comparison tests whether **where measurements are allocated** matters beyond **how many measurements are used**. It should not be substituted for a quantum-versus-classical model-capacity comparison.

---

## Zero-Trust Integration

The scorer supplies **off-path evidence**, not autonomous access-control authority. Adaptive measurement demand can affect later evidence updates without making the QNN an intermediate hop for the current TI request.

### Evidence Plane

```text
Mirrored Aggregate-Flow Telemetry
              |
Train-Derived Feature Preparation
              |
Classical Embedder → Quantum Encoding
              |
VQC Measurements: 128 Cumulative Shots
              |
Classical Head → Distance to Fixed Threshold
              |
Outside Calibrated Margin? ── Yes ──> Return Evidence
              |
              No
              |
Add Shots → 256 → 512 → 1024 Maximum
              |
Return Evidence and Measurement Budget
              |
Contextual Policy Process → Subsequent Updates
```

### Enforcement Plane

```text
Current TI Request
       |
Policy Enforcement Point (PEP)
       |
Cached Deterministic Policy
       |
Grant / Restrict / Step-Up / Deny
       |
Protected Tactile Internet Service
```

This is the architectural interpretation of the scorer. The adaptive-shot results do not measure a full PDP/PEP deployment or prove a haptic-loop latency budget.

---

## Intended Uses

The framework supports research on:

- adaptive measurement allocation for threshold-based hybrid QNN inference;
- finite-shot ranking, probability quality, and decision stability;
- leakage-safe network anomaly evaluation;
- matched-budget experimental controls;
- off-path TI-security evidence generation; and
- reproducibility of the associated paper's measurement–reliability study.

---

## Out-of-Scope Uses

The model should **not** be treated as:

- a production intrusion detector or verified attack detector;
- a calibrated probability-of-compromise estimator;
- an autonomous access-control authority;
- proof of quantum advantage or physical-hardware robustness;
- evidence of end-to-end TI latency compliance; or
- a formal test-reliability guarantee implied by the AS-VQC-95 name.

Security-critical actions require independent evidence, suitable policy safeguards, and operational validation.

---

## Limitations

1. **Statistical labels and proxy data:** quantile-derived labels and CESNET aggregates do not establish verified intrusion detection or dedicated TI-traffic performance.
2. **Ideal measurement sampling:** finite-shot effects are evaluated, but device noise, readout errors, transpilation, queueing, and physical execution costs are not.
3. **Empirical calibration:** validation-error percentiles are not formal test-time coverage guarantees, particularly under distribution shift.
4. **Reliability tradeoff:** Fixed-1024 is more decision-stable than AS-VQC-95 despite the latter's measurement savings.
5. **Checkpoint variability:** Group seed 46 substantially increases mean measurement demand and remains included in reporting.
6. **Statistical scope:** five checkpoints per holdout are the model-level units; measurement repetitions do not increase that count.
7. **Application scope:** the stopping rule requires a scalar probabilistic output and a usable threshold; transfer to other quantum algorithms is not established.
8. **Deployment scope:** shot reductions do not directly measure wall-clock, energy, security-effectiveness, or end-to-end TI benefits.

See the [paper's Discussion and Limitations](https://arxiv.org/html/2610.05835v1#S5) for the interpretation of these boundaries.

---

## Reproducibility

**GitHub:**  
https://github.com/msudipto/AdaptiveShot_HybridQAD_Framework

The repository documents preprocessing, split construction, QNN training, adaptive calibration, finite-shot evaluation, aggregation, and figure generation. Settings are under [config/adaptive_shots.yaml](config/adaptive_shots.yaml); data placement is documented in [data/README.md](data/README.md).

```text
AdaptiveShot_HybridQAD_Framework/
├── artifacts/
├── config/
├── data/
├── src/
├── LICENSE
├── README.md
├── commands.txt
├── requirements.txt
├── run_pipeline.ps1
└── run_adaptive-shot_hybrid-qad.ps1
```

### Repository Installation

Clone the **adaptive-shot repository**:

```bash
git clone https://github.com/msudipto/AdaptiveShot_HybridQAD_Framework.git
cd AdaptiveShot_HybridQAD_Framework
```

On Windows, create a separate Python 3.12 environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the required CESNET data according to `data/README.md` and check configuration paths before starting the documented workflow:

```powershell
.\run_adaptive-shot_hybrid-qad.ps1
```

The repository also documents `run_pipeline.ps1` as the equivalent pipeline entry point. The full workflow can generate missing checkpoints and experimental artifacts; it is not merely a download command.

For Unix-like environments, environment setup is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The supplied full-pipeline wrappers are PowerShell scripts. Consult `commands.txt` for the individual Python stages rather than assuming a Bash wrapper is included.

To generate composite figures from existing result artifacts without retraining:

```bash
python -m src.make_combined_figures
```

### Artifact and Execution Scope

The linked repository documents lightweight summaries and publication outputs under `artifacts/`; large checkpoints, raw data, repeated per-record realizations, and caches are excluded by default unless intentionally archived. Do not assume a clone includes pretrained weights or the raw CESNET dataset.

The paper reports CPU-based analytic training and finite-shot simulation. These README commands follow the documented repository workflow; they were **not executed as part of this documentation update**.

---

## Downloading from Hugging Face

The supplied Hugging Face link is the **paper page**:

https://huggingface.co/papers/2610.05835

It is not a model or dataset repository identifier. No separate downloadable checkpoint-repository ID is specified here; a model-weight download command would require that additional identifier. Use the linked GitHub framework for the documented reconstruction and experiment workflow.

---

## Paper

**Mubassir Serneabat Sudipto, Shakil Ahmed, Ashfaq Khokhar, and Samir M. Iqbal.**  
**“Adaptive-Shot Hybrid Quantum Anomaly Detection for Tactile Internet Security: Reliability-Aware Measurement Allocation Under Resource Constraints.”**  
arXiv:2610.05835, 2026.

The arXiv comments report acceptance for a **Short Oral Presentation at the NeurIPS 2026 SaTQuML Workshop**. This is a workshop designation, not a claim of acceptance in the NeurIPS main-conference track.

- **arXiv:** https://arxiv.org/abs/2610.05835
- **Full text:** https://arxiv.org/html/2610.05835v1
- **Hugging Face Papers:** https://huggingface.co/papers/2610.05835
- **DOI:** https://doi.org/10.48550/arXiv.2610.05835
- **Code:** https://github.com/msudipto/AdaptiveShot_HybridQAD_Framework

---

## Citation

Please cite the paper when using its methodology or reported results, and cite the repository when using its software.

### Paper Citation

```bibtex
@misc{sudipto2026adaptiveshothybridquantum,
  title         = {Adaptive-Shot Hybrid Quantum Anomaly Detection for Tactile Internet Security: Reliability-Aware Measurement Allocation Under Resource Constraints},
  author        = {Sudipto, Mubassir Serneabat and Ahmed, Shakil and Khokhar, Ashfaq and Iqbal, Samir M.},
  year          = {2026},
  eprint        = {2610.05835},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  doi           = {10.48550/arXiv.2610.05835},
  url           = {https://arxiv.org/abs/2610.05835},
  note          = {Accepted at the NeurIPS 2026 Secure and Trustworthy Quantum Machine Learning (SaTQuML) Workshop; Short Oral Presentation}
}
```

### Software Repository Citation

```bibtex
@misc{adaptive_shot_hybrid_qad_framework_2026,
  author       = {Sudipto, Mubassir Serneabat and Ahmed, Shakil and Khokhar, Ashfaq and Iqbal, Samir M.},
  title        = {{Adaptive-Shot Hybrid QAD Framework}: Reliability-Aware Finite-Shot Hybrid Quantum Anomaly Detection for Tactile Internet Security},
  year         = {2026},
  howpublished = {\url{https://github.com/msudipto/AdaptiveShot_HybridQAD_Framework}},
  note         = {Code repository},
  url          = {https://github.com/msudipto/AdaptiveShot_HybridQAD_Framework}
}
```

---

## License

The linked **AdaptiveShot_HybridQAD_Framework software is released under the MIT License**. See [LICENSE](LICENSE) for the repository's terms.

The [arXiv manuscript](https://arxiv.org/html/2610.05835v1) is available under **CC BY 4.0**. CESNET-TimeSeries24 data retain their own attribution and license requirements; the software license does not replace those terms.

---

## Acknowledgment

This research supports reproducible work on trustworthy quantum machine learning, adaptive finite-shot inference, network anomaly evidence, and Tactile-Internet-oriented security.

The project emphasizes transparent measurement accounting and careful separation between statistical-benchmark results, ideal quantum sampling, and claims of operational security or hardware advantage. Dataset attribution and broader-impact considerations are documented in the associated paper.

---

## Contact

**Mubassir Serneabat Sudipto**  
Iowa State University  
[msudipto@iastate.edu](mailto:msudipto@iastate.edu)

**Shakil Ahmed**  
Grand Valley State University  
[ahmeshak@gvsu.edu](mailto:ahmeshak@gvsu.edu)

**Ashfaq Khokhar**  
Kansas State University  
[akhokhar@ksu.edu](mailto:akhokhar@ksu.edu)

**Samir M. Iqbal**  
Grand Valley State University  
[iqbalsa@gvsu.edu](mailto:iqbalsa@gvsu.edu)

---

**Paper:** https://arxiv.org/abs/2610.05835  
**Hugging Face Paper:** https://huggingface.co/papers/2610.05835  
**GitHub:** https://github.com/msudipto/AdaptiveShot_HybridQAD_Framework
