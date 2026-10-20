## Description
Provide a concise overview of the optimization tweaks, circuit changes, or statistical stopping rule adjustments introduced by this PR.

## Related Issue
Fixes # (link your custom Bug Report or Feature Request issue number here)

## Change Type
- [ ] Bug fix (non-breaking change fixing simulation loops, feature scaling math, or graph pipeline blocks)
- [ ] New feature (non-breaking extension adding adaptive shot thresholds or alternative ansatz configurations)
- [ ] Refactoring / Documentation alignment

## Simulation & Policy Impact Checklist
Please verify how your modifications behave relative to the framework's baseline evaluation setup:
- [ ] My modifications preserve the validation-error percentile calibration rules for **AS-VQC-95** (or alternative AS-VQC policies).
- [ ] The model successfully handles the **12 payload-independent aggregate-flow features** from the tabular dataset without shape errors.
- [ ] I have verified that this change does not break baseline comparison logic (Fixed-Shot references or the **Budget-Shuffled-AS95** randomized control layout).

## Local Pipeline Execution Verification
- [ ] I have executed the primary PowerShell tracking wrappers (`.\run_adaptive-shot_hybrid-qad.ps1` or `.\run_pipeline.ps1`) locally, and they complete successfully.
- [ ] Standing visualization tasks like `python -m src.make_combined_figures` run cleanly and render stable assets into the `artifacts/` directory.

## Git and Artifact Containment Integrity
- [ ] I have **NOT** included heavy training weights checkpoints, duplicate processing caches, or raw tabular dataset blocks in this tracking commit.
