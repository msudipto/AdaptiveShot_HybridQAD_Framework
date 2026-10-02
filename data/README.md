# Data Directory

This directory documents how CESNET-TimeSeries24 data should be organized for local experimentation and reproducible evaluation.

## Purpose

The `data/` directory is intended for dataset placement instructions, local raw/processed data organization, lightweight public samples when permitted, and reproducibility notes. Raw third-party data are not redistributed by this repository.

## Recommended Layout

```text
data/
├── README.md
├── raw/
│   └── cesnet_timeseries24/
│       └── ip_addresses_sample/
│           └── agg_10_minutes/
└── processed/
    ├── cesnet_raw_candidates_4875.npz
    └── splits/
```

## Expected Workflow

1. Obtain CESNET-TimeSeries24 from its original source.
2. Place the 10-minute IP-address aggregate files under `data/raw/cesnet_timeseries24/ip_addresses_sample/agg_10_minutes/`.
3. Run `python -m src.prepare_candidate --config config/adaptive_shots.yaml`.
4. Build leakage-safe splits with `python -m src.make_splits --config config/adaptive_shots.yaml`.
5. Continue with training/evaluation or use `run_pipeline.ps1` for the full workflow.

## Data Sharing Policy

Do not commit raw CESNET data, restricted datasets, private data, or large regenerable processed files. Follow the original dataset license and terms of use.

## Reproducibility Guidance

The manuscript-facing configuration fixes the 4,875-record candidate sample, feature set, split protocols, random seeds, train-only preprocessing, and pseudo-label construction in `config/adaptive_shots.yaml`.
