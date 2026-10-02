"""Generate publication-ready Adaptive-Shot Hybrid QAD figures from completed adaptive-shot artifacts.

This module is intentionally read-only with respect to experimental results.
It consumes the hierarchical aggregation outputs plus saved primary
AS-VQC-95 allocations and writes PDF/PNG figures and a figure manifest.

Run directly:
    python -m src.make_figures --config config/adaptive_shots.yaml

The main orchestration script run_adaptive-shot_hybrid-qad.ps1 invokes this module automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROTOCOLS = ["random", "group", "temporal"]
PROTOCOL_LABEL = {
    "random": "Random",
    "group": "Group",
    "temporal": "Temporal",
}

FIXED_POLICIES = [
    "Fixed-128",
    "Fixed-256",
    "Fixed-512",
    "Fixed-1024",
]

ADAPTIVE_POLICIES = [
    "AS-VQC-90",
    "AS-VQC-95",
    "AS-VQC-99",
]


# ============================================================
# Publication Style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.4,
    "axes.labelsize": 8.4,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.30,
    "lines.markersize": 4.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.pad_inches": 0.03,
})


def clean_axis(ax, ygrid=True):
    """Minimal conference-style axis formatting."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ygrid:
        ax.grid(axis="y", linewidth=0.45, alpha=0.24)

    ax.tick_params(length=3, width=0.65)


def add_panel_label(ax, label):
    """Add publication-style subfigure label."""
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
    )


def save_figure(fig, outdir: Path, stem: str):
    """Write vector PDF and high-resolution PNG."""
    outdir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        outdir / f"{stem}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        outdir / f"{stem}.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


def load_results(root: Path):
    summary_dir = root / "artifacts" / "summary"
    figures_dir = root / "artifacts" / "figures"

    summary_path = summary_dir / "summary_mean_sd.csv"
    seed_path = summary_dir / "seed_level_metrics.csv"
    paired_path = summary_dir / "paired_bootstrap.csv"
    distance_path = figures_dir / "threshold_distance_summary.csv"

    required = [
        summary_path,
        seed_path,
        paired_path,
        distance_path,
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required Adaptive-Shot Hybrid QAD artifact(s):\n  "
            + "\n  ".join(missing)
            + "\nRun the completed Adaptive-Shot Hybrid QAD pipeline first."
        )

    summary = pd.read_csv(summary_path)
    seed = pd.read_csv(seed_path)
    paired = pd.read_csv(paired_path)
    distance = pd.read_csv(distance_path)

    return summary, seed, paired, distance


def protocol_summary(summary: pd.DataFrame, protocol: str):
    return summary[summary["Protocol"] == protocol].set_index("Policy")


# ============================================================
# Figure 1
# Ranking + Reliability
# ============================================================

def make_figure_1(summary, outdir):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.85, 2.75),
    )

    ax_a, ax_b = axes

    # --------------------------------------------------------
    # (a) AUC deviation from analytic inference
    # --------------------------------------------------------

    protocol_markers = {
        "random": "o",
        "group": "s",
        "temporal": "^",
    }

    for protocol in PROTOCOLS:

        s = protocol_summary(summary, protocol)
        analytic_auc = float(s.loc["Analytic", "AUC_Mean"])

        fixed_x = np.array([
            float(s.loc[p, "AverageShots_Mean"])
            for p in FIXED_POLICIES
        ])

        fixed_y = 1e4 * np.array([
            analytic_auc - float(s.loc[p, "AUC_Mean"])
            for p in FIXED_POLICIES
        ])

        adaptive_x = np.array([
            float(s.loc[p, "AverageShots_Mean"])
            for p in ADAPTIVE_POLICIES
        ])

        adaptive_y = 1e4 * np.array([
            analytic_auc - float(s.loc[p, "AUC_Mean"])
            for p in ADAPTIVE_POLICIES
        ])

        marker = protocol_markers[protocol]

        ax_a.plot(
            fixed_x,
            fixed_y,
            marker=marker,
            label=f"{PROTOCOL_LABEL[protocol]} Fixed",
        )

        ax_a.plot(
            adaptive_x,
            adaptive_y,
            marker=marker,
            linestyle="--",
            label=f"{PROTOCOL_LABEL[protocol]} Adaptive",
        )

    ax_a.axhline(
        0,
        linewidth=0.7,
        alpha=0.7,
    )

    ax_a.set_xscale(
        "log",
        base=2,
    )

    ax_a.set_xticks(
        [128, 256, 512, 1024],
        ["128", "256", "512", "1024"],
    )

    ax_a.set_xlabel(
        "Average Shots Per Record"
    )

    ax_a.set_ylabel(
        r"AUC Loss vs. Analytic ($\times10^{-4}$)"
    )

    clean_axis(ax_a)
    add_panel_label(ax_a, "(a)")

    # --------------------------------------------------------
    # (b) Reliability–measurement frontier
    # --------------------------------------------------------

    for protocol in PROTOCOLS:

        s = protocol_summary(summary, protocol)

        fixed_x = np.array([
            float(s.loc[p, "AverageShots_Mean"])
            for p in FIXED_POLICIES
        ])

        fixed_y = 100 * np.array([
            float(s.loc[p, "DecisionDisagreement_Mean"])
            for p in FIXED_POLICIES
        ])

        adaptive_x = np.array([
            float(s.loc[p, "AverageShots_Mean"])
            for p in ADAPTIVE_POLICIES
        ])

        adaptive_y = 100 * np.array([
            float(s.loc[p, "DecisionDisagreement_Mean"])
            for p in ADAPTIVE_POLICIES
        ])

        marker = protocol_markers[protocol]

        ax_b.plot(
            fixed_x,
            fixed_y,
            marker=marker,
            label=f"{PROTOCOL_LABEL[protocol]} Fixed",
        )

        ax_b.plot(
            adaptive_x,
            adaptive_y,
            marker=marker,
            linestyle="--",
            label=f"{PROTOCOL_LABEL[protocol]} Adaptive",
        )

        # Primary AS-VQC-95 operating point.
        ax_b.scatter(
            [adaptive_x[1]],
            [adaptive_y[1]],
            marker="*",
            s=58,
            zorder=6,
        )

    ax_b.set_xscale(
        "log",
        base=2,
    )

    ax_b.set_xticks(
        [128, 256, 512, 1024],
        ["128", "256", "512", "1024"],
    )

    ax_b.set_xlabel(
        "Average Shots Per Record"
    )

    ax_b.set_ylabel(
        "Decision Disagreement (%)"
    )

    clean_axis(ax_b)
    add_panel_label(ax_b, "(b)")

    # Compact shared legend.
    handles, labels = ax_b.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        columnspacing=1.0,
        handletextpad=0.4,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        bottom=0.20,
        top=0.78,
        wspace=0.28,
    )

    save_figure(
        fig,
        outdir,
        "fig1_ranking_and_reliability",
    )


# ============================================================
# Figure 2
# Shot Allocation + Boundary Proximity
# ============================================================

def make_figure_2(summary, distance, outdir):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.85, 2.85),
    )

    ax_a, ax_b = axes

    # --------------------------------------------------------
    # (a) AS-VQC-95 shot allocation
    # --------------------------------------------------------

    as95 = (
        summary[
            summary["Policy"] == "AS-VQC-95"
        ]
        .set_index("Protocol")
        .loc[PROTOCOLS]
    )

    shot_columns = [
        ("FracShots128_Mean", "128"),
        ("FracShots256_Mean", "256"),
        ("FracShots512_Mean", "512"),
        ("FracShots1024_Mean", "1024"),
    ]

    y = np.arange(
        len(PROTOCOLS)
    )

    left = np.zeros(
        len(PROTOCOLS)
    )

    for column, label in shot_columns:

        values = (
            100
            * as95[column]
            .to_numpy(dtype=float)
        )

        bars = ax_a.barh(
            y,
            values,
            left=left,
            height=0.52,
            label=f"{label} Shots",
        )

        for yi, value, left_value in zip(
            y,
            values,
            left,
        ):

            if value >= 4.0:

                ax_a.text(
                    left_value + value / 2,
                    yi,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=6.7,
                )

        left += values

    ax_a.set_yticks(
        y,
        [PROTOCOL_LABEL[p] for p in PROTOCOLS],
    )

    ax_a.invert_yaxis()

    ax_a.set_xlim(
        0,
        100,
    )

    ax_a.set_xlabel(
        "Final Shot Allocation (%)"
    )

    ax_a.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.50, 1.01),
        columnspacing=0.9,
        handletextpad=0.35,
    )

    clean_axis(ax_a)
    add_panel_label(ax_a, "(a)")

    # --------------------------------------------------------
    # (b) Threshold distance vs final shot count
    # --------------------------------------------------------

    shot_levels = [
        128,
        256,
        512,
        1024,
    ]

    xbase = np.arange(
        len(shot_levels),
        dtype=float,
    )

    offsets = [
        -0.14,
        0.0,
        0.14,
    ]

    markers = [
        "o",
        "s",
        "^",
    ]

    for offset, marker, protocol in zip(
        offsets,
        markers,
        PROTOCOLS,
    ):

        d = (
            distance[
                distance["Protocol"] == protocol
            ]
            .set_index("FinalShots")
            .loc[shot_levels]
        )

        median = d[
            "MedianDistance"
        ].to_numpy(dtype=float)

        q1 = d[
            "Q1"
        ].to_numpy(dtype=float)

        q3 = d[
            "Q3"
        ].to_numpy(dtype=float)

        ax_b.errorbar(
            xbase + offset,
            median,
            yerr=np.vstack([
                median - q1,
                q3 - median,
            ]),
            fmt=marker,
            linestyle="none",
            capsize=2.3,
            label=PROTOCOL_LABEL[protocol],
        )

    ax_b.set_yscale(
        "log",
    )

    ax_b.set_xticks(
        xbase,
        [str(s) for s in shot_levels],
    )

    ax_b.set_xlabel(
        "Final Allocated Shots"
    )

    ax_b.set_ylabel(
        r"Distance to Threshold $|p-\tau^*|$"
    )

    ax_b.legend(
        frameon=False,
        loc="upper right",
    )

    clean_axis(ax_b)
    add_panel_label(ax_b, "(b)")

    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        bottom=0.20,
        top=0.87,
        wspace=0.30,
    )

    save_figure(
        fig,
        outdir,
        "fig2_allocation_and_boundary",
    )


# ============================================================
# Figure 3
# Targeted Allocation + TI Operating Points
# ============================================================

def make_figure_3(summary, paired, outdir):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.85, 2.80),
    )

    ax_a, ax_b = axes

    # --------------------------------------------------------
    # (a) Targeted vs matched-budget shuffled control
    # --------------------------------------------------------

    cmp_df = paired[
        (paired["PolicyA"] == "AS-VQC-95")
        & (
            paired["PolicyB"]
            == "Budget-Shuffled-AS95"
        )
        & (
            paired["Metric"]
            == "DecisionDisagreement"
        )
    ].set_index("Protocol").loc[PROTOCOLS]

    effect = (
        -100
        * cmp_df[
            "MeanDifference_AminusB"
        ].to_numpy(dtype=float)
    )

    lower = (
        -100
        * cmp_df[
            "CI95_Hi"
        ].to_numpy(dtype=float)
    )

    upper = (
        -100
        * cmp_df[
            "CI95_Lo"
        ].to_numpy(dtype=float)
    )

    xerr = np.vstack([
        effect - lower,
        upper - effect,
    ])

    yy = np.arange(
        len(PROTOCOLS)
    )

    ax_a.errorbar(
        effect,
        yy,
        xerr=xerr,
        fmt="o",
        capsize=3,
        markersize=5.2,
        linestyle="none",
    )

    ax_a.axvline(
        0,
        linewidth=0.8,
    )

    ax_a.set_yticks(
        yy,
        [PROTOCOL_LABEL[p] for p in PROTOCOLS],
    )

    ax_a.invert_yaxis()

    ax_a.set_xlabel(
        "Reduction in Decision Disagreement\n(Percentage Points)"
    )

    clean_axis(
        ax_a,
        ygrid=False,
    )

    ax_a.grid(
        axis="x",
        linewidth=0.45,
        alpha=0.24,
    )

    add_panel_label(
        ax_a,
        "(a)",
    )

    # --------------------------------------------------------
    # (b) TPR/FPR operating behavior
    # --------------------------------------------------------

    policies = [
        "Fixed-128",
        "AS-VQC-95",
        "AS-VQC-99",
        "Fixed-1024",
    ]

    policy_markers = {
        "Fixed-128": "o",
        "AS-VQC-95": "*",
        "AS-VQC-99": "D",
        "Fixed-1024": "s",
    }

    for protocol in PROTOCOLS:

        s = protocol_summary(
            summary,
            protocol,
        )

        x = 100 * np.array([
            float(s.loc[p, "FPR_Mean"])
            for p in policies
        ])

        y = 100 * np.array([
            float(s.loc[p, "TPR_Mean"])
            for p in policies
        ])

        ax_b.plot(
            x,
            y,
            linewidth=0.9,
            alpha=0.7,
            label=PROTOCOL_LABEL[protocol],
        )

        for xv, yv, policy in zip(
            x,
            y,
            policies,
        ):

            size = (
                55
                if policy == "AS-VQC-95"
                else 27
            )

            ax_b.scatter(
                [xv],
                [yv],
                marker=policy_markers[policy],
                s=size,
                zorder=5,
            )

    ax_b.set_xlabel(
        "False Positive Rate (%)"
    )

    ax_b.set_ylabel(
        "True Positive Rate (%)"
    )

    ax_b.legend(
        frameon=False,
        loc="best",
    )

    clean_axis(ax_b)
    add_panel_label(ax_b, "(b)")

    fig.subplots_adjust(
        left=0.11,
        right=0.995,
        bottom=0.22,
        top=0.94,
        wspace=0.35,
    )

    save_figure(
        fig,
        outdir,
        "fig3_targeting_and_operating_points",
    )


# ============================================================
# Figure 4
# Calibration + Checkpoint Variability
# ============================================================

def make_figure_4(summary, seed, outdir):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.85, 2.80),
    )

    ax_a, ax_b = axes

    # --------------------------------------------------------
    # (a) Calibration vs average shots
    # --------------------------------------------------------

    protocol_markers = {
        "random": "o",
        "group": "s",
        "temporal": "^",
    }

    for protocol in PROTOCOLS:

        s = protocol_summary(
            summary,
            protocol,
        )

        fixed_x = np.array([
            float(
                s.loc[
                    p,
                    "AverageShots_Mean",
                ]
            )
            for p in FIXED_POLICIES
        ])

        fixed_y = 100 * np.array([
            float(
                s.loc[
                    p,
                    "ECE_Mean",
                ]
            )
            for p in FIXED_POLICIES
        ])

        adaptive_x = np.array([
            float(
                s.loc[
                    p,
                    "AverageShots_Mean",
                ]
            )
            for p in ADAPTIVE_POLICIES
        ])

        adaptive_y = 100 * np.array([
            float(
                s.loc[
                    p,
                    "ECE_Mean",
                ]
            )
            for p in ADAPTIVE_POLICIES
        ])

        marker = protocol_markers[
            protocol
        ]

        ax_a.plot(
            fixed_x,
            fixed_y,
            marker=marker,
            label=f"{PROTOCOL_LABEL[protocol]} Fixed",
        )

        ax_a.plot(
            adaptive_x,
            adaptive_y,
            marker=marker,
            linestyle="--",
            label=f"{PROTOCOL_LABEL[protocol]} Adaptive",
        )

    ax_a.set_xscale(
        "log",
        base=2,
    )

    ax_a.set_xticks(
        [128, 256, 512, 1024],
        ["128", "256", "512", "1024"],
    )

    ax_a.set_xlabel(
        "Average Shots Per Record"
    )

    ax_a.set_ylabel(
        "Expected Calibration Error (%)"
    )

    clean_axis(ax_a)
    add_panel_label(ax_a, "(a)")

    # --------------------------------------------------------
    # (b) Seed-wise AS-VQC-95 resource demand
    # --------------------------------------------------------

    as95_seed = seed[
        seed["Policy"]
        == "AS-VQC-95"
    ].copy()

    seed_markers = {
        "random": "o",
        "group": "s",
        "temporal": "^",
    }

    for protocol in PROTOCOLS:

        d = (
            as95_seed[
                as95_seed["Protocol"]
                == protocol
            ]
            .sort_values(
                "TrainingSeed"
            )
        )

        ax_b.plot(
            d[
                "TrainingSeed"
            ].to_numpy(dtype=int),
            d[
                "AverageShots"
            ].to_numpy(dtype=float),
            marker=seed_markers[
                protocol
            ],
            label=PROTOCOL_LABEL[
                protocol
            ],
        )

    ax_b.set_yscale(
        "log",
        base=2,
    )

    ax_b.set_yticks(
        [128, 256, 512, 1024],
        ["128", "256", "512", "1024"],
    )

    ax_b.set_xticks(
        [42, 43, 44, 45, 46]
    )

    ax_b.set_xlabel(
        "Training Seed"
    )

    ax_b.set_ylabel(
        "Average Shots Per Record"
    )

    ax_b.legend(
        frameon=False,
        loc="upper left",
    )

    # Explicitly identify high-escalation checkpoint.
    group46 = as95_seed[
        (as95_seed["Protocol"] == "group")
        & (
            as95_seed["TrainingSeed"]
            == 46
        )
    ]

    if not group46.empty:

        row = group46.iloc[0]

        ax_b.annotate(
            f"Group-46\n{row['AverageShots']:.1f} Shots",
            (
                row["TrainingSeed"],
                row["AverageShots"],
            ),
            xytext=(-52, -7),
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 0.65,
            },
            fontsize=6.8,
        )

    clean_axis(ax_b)
    add_panel_label(ax_b, "(b)")

    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        bottom=0.20,
        top=0.94,
        wspace=0.33,
    )

    save_figure(
        fig,
        outdir,
        "fig4_calibration_and_resource_variability",
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=".",
        help=(
            "Repository root. Default: current directory."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "figures_combined"
        ),
        help=(
            "Directory for combined figures."
        ),
    )

    args = parser.parse_args()

    root = Path(
        args.root
    ).resolve()

    outdir = (
        root
        / args.output_dir
    )

    summary, seed, paired, distance = load_results(
        root
    )

    make_figure_1(
        summary,
        outdir,
    )

    make_figure_2(
        summary,
        distance,
        outdir,
    )

    make_figure_3(
        summary,
        paired,
        outdir,
    )

    make_figure_4(
        summary,
        seed,
        outdir,
    )

    print()
    print(
        "[combined_figures] "
        "Generated 4 composite figures."
    )

    print(
        "[combined_figures] "
        "Each figure contains 2 side-by-side panels."
    )

    print(
        "[combined_figures] "
        f"Output: {outdir}"
    )

    print()

    for name in [
        "fig1_ranking_and_reliability",
        "fig2_allocation_and_boundary",
        "fig3_targeting_and_operating_points",
        "fig4_calibration_and_resource_variability",
    ]:

        print(
            "  - "
            + name
            + ".pdf"
        )


if __name__ == "__main__":
    main()