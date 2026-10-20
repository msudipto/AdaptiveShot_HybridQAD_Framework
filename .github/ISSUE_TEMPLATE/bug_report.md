---
name: Bug report
about: Report script crashes, feature leakages, or finite-shot simulation errors in AS-VQC
title: '[BUG] '
labels: 'bug'
assignees: ''

---

**Describe the bug**
A clear and concise description of the crash, script execution failure, or unexpected results mismatch.

**To Reproduce**
Steps to reproduce the behavior:
1. Run pipeline script `.\run_pipeline.ps1` or `.\run_adaptive-shot_hybrid-qad.ps1`
2. Configuration parameter used in `config/adaptive_shots.yaml`
3. See error output log

**Expected behavior**
A clear description of the expected behavior (e.g., successful validation calibration, correct bootstrap resample array lengths).

**Simulation & Pipeline Context (please complete if applicable):**
 - Python Module / Script: [e.g., src.training, src.make_combined_figures]
 - Holdout Split Method: [e.g., Random Holdout, Entity-Group, Temporal]
 - Running Policy: [e.g., AS-VQC-90, AS-VQC-95, AS-VQC-99, Budget-Shuffled-AS95]
 - Terminal Stack Trace / Error Logs:
   ```text
   Paste tracebacks or console errors here
   ```

**Additional context**
Add any other context about seed values (42–46), CESNET tensor sizes, or environment anomalies.
