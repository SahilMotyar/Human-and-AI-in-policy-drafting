"""
regenerate_figures.py
=====================
Re-renders the three figures whose layouts had rendering bugs, reading straight
from the saved master-data CSVs so the underlying data and statistics are
IDENTICAL to the original analysis pipeline. Only the *layout* is corrected:

  * Fig01  - the "Ideal baseline origin" annotation was clipped off the right
             edge (it ran into the marginal KDE axis).
  * Fig02  - each panel's italic subtitle overlapped its bold panel title.
  * Fig03  - the legend box sat directly on top of the boxplot data.

The full analysis pipelines live in `semantic.py` (Fig01) and
`complexity_compression.py` (Fig02 / Fig03); those scripts carry the same layout
fixes. This helper exists so the figures can be regenerated from the saved
metrics without re-running the heavy NLP/embedding pipeline.

Data sources:
    result/final_outputs/04_master_data.csv          (Fig01)
    result/master_dataframe_with_stage3_scores.csv   (Fig02, Fig03)

Run:  python regenerate_figures.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.stats import spearmanr

RESULT_DIR = "result"
FINAL_OUT_DIR = os.path.join(RESULT_DIR, "final_outputs")
FIG_DIR = os.path.join(FINAL_OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MASTER_CSV = os.path.join(FINAL_OUT_DIR, "04_master_data.csv")
STAGE3_CSV = os.path.join(RESULT_DIR, "master_dataframe_with_stage3_scores.csv")

COND_ORDER = ["Status Quo", "Innovation", "Unconstrained"]
COND_COLORS = {
    "Status Quo":    "#534AB7",
    "Innovation":    "#0F6E56",
    "Unconstrained": "#639922",
}


# --------------------------------------------------------------------------- #
# Fig01 - Stage 2 semantic vs lexical drift (JointGrid scatter)
# --------------------------------------------------------------------------- #
def plot_confidence_ellipse(ax, x, y, color, n_std=1.5, alpha=0.15):
    """1.5-sigma covariance ellipse, identical to semantic.py."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)), width=width, height=height, angle=angle,
        facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.5, linestyle="--",
    )
    ax.add_patch(ellipse)


def make_fig01():
    d = pd.read_csv(MASTER_CSV)
    d["is_partial"] = (
        d["is_partial"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    )
    primary_df = d[
        (d["corpus_type"] == "Primary")
        & (~d["is_partial"])
        & (~d["model"].str.contains("sarvam", case=False, na=False))
    ].copy()
    print(f"  Fig01: {len(primary_df)} primary drafts")

    g = sns.JointGrid(data=primary_df, x="bert_score_f1", y="jsd_score", height=9, ratio=5)
    g.plot_joint(
        sns.scatterplot, hue=primary_df["condition"], palette=COND_COLORS,
        style=primary_df["domain"], s=120, alpha=0.8, edgecolor="white",
    )
    g.plot_marginals(
        sns.kdeplot, hue=primary_df["condition"], palette=COND_COLORS,
        fill=True, alpha=0.25, linewidth=1.5,
    )

    ax = g.ax_joint
    x_origin = primary_df["bert_score_f1"].max() + 0.02
    y_origin = primary_df["jsd_score"].min() - 0.02

    for cond, col in COND_COLORS.items():
        sub = primary_df[primary_df["condition"] == cond]
        plot_confidence_ellipse(ax, sub["bert_score_f1"], sub["jsd_score"], col)
        mx, my = sub["bert_score_f1"].mean(), sub["jsd_score"].mean()
        ax.annotate("", xy=(mx, my), xytext=(x_origin, y_origin),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.8, alpha=0.6))

    ax.scatter(1.0, 0.0, color="#E24B4A", marker="*", s=600, edgecolor="black",
               label="Baseline Origin (1.0, 0.0)", zorder=5)

    # LAYOUT FIX: original xytext=(0.75, 0.10) pushed the text box off the right
    # edge into the marginal axis, clipping it. Center it further left so the
    # whole box stays inside the joint axes while still pointing at the star.
    ax.annotate(
        "Ideal baseline\norigin (1.0, 0.0)",
        xy=(1.0, 0.0), xytext=(0.58, 0.13),
        xycoords="data", textcoords="axes fraction", ha="center",
        arrowprops=dict(arrowstyle="->", color="#E24B4A", lw=1.5),
        fontsize=9, color="#E24B4A",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E24B4A", alpha=0.85),
    )

    r, p = spearmanr(primary_df["bert_score_f1"], primary_df["jsd_score"])
    ax.text(0.02, 0.97, f"Spearman r = {r:.3f} (p={p:.3f})", transform=ax.transAxes,
            fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    ax.set_xlim(primary_df["bert_score_f1"].min() - 0.05, 1.05)
    ax.set_ylim(-0.05, primary_df["jsd_score"].max() + 0.05)

    plt.suptitle("Stage 2 - Algorithmic Drift: Baseline vs Semantic Drift", fontsize=16, y=1.02)
    ax.set_xlabel("Semantic Similarity (BERTScore F1) $\\rightarrow$", fontsize=12)
    ax.set_ylabel("Lexical Drift (Jensen-Shannon Divergence) $\\rightarrow$", fontsize=12)

    joint_legend = ax.get_legend()
    if joint_legend is not None:
        handles, labels = ax.get_legend_handles_labels()
        joint_legend.remove()
        g.fig.legend(handles, labels, bbox_to_anchor=(1.15, 0.85), loc="upper right",
                     fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.6)

    out = os.path.join(FIG_DIR, "Fig01_Stage2_Semantic_Lexical_Drift.png")
    g.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  saved {out}")


# --------------------------------------------------------------------------- #
# Fig02 / Fig03 - Stage 3 complexity compression
# --------------------------------------------------------------------------- #
MODEL_ORDER = ["ChatGPT", "Gemini"]

# Statute grand-mean reference values (avg. of the 3 baseline statutes), matching
# the labels printed on the original figure by complexity_compression.py.
BASELINE_GRAND = {
    "mattr":         0.70,
    "hedge_density": 0.46,
    "mean_sent_len": 36.62,
    "sent_len_var":  1691.19,
}
BASELINE_ABOVE_DATA = {
    "mattr":         False,   # statute MATTR < AI drafts
    "hedge_density": True,    # statute hedge density > AI drafts
    "mean_sent_len": True,    # statute sentences longer than AI drafts
    "sent_len_var":  True,    # statute variance >= AI drafts (except Status Quo)
}
METRICS = [
    ("mattr",
     "MATTR  (Lexical Diversity)",
     "MATTR score",
     "Higher = more lexically diverse per 50-word window",
     "Statute MATTR lower than AI\n(statutes repeat defined terms heavily)"),
    ("hedge_density",
     "Hedge Density  (Conjunctions / 100 words)",
     "Hedges per 100 words",
     "Higher = more conditional legal architecture preserved",
     "AI drafts fall short of statute\nhedge density across all conditions"),
    ("mean_sent_len",
     "Mean Sentence Length  (words)",
     "Words per sentence",
     "Higher = longer, more nested operative clauses",
     "AI sentences shorter than statute\n(operative clause compression)"),
    ("sent_len_var",
     "Sentence Length Variance",
     "Variance  (words²)",
     "Higher = wider mix of short definitional + long operative sentences",
     "Status Quo closest to statute variance;\nUnconstrained collapses toward uniformity"),
]
BOX_W, WITHIN_GAP, GROUP_GAP = 0.55, 0.72, 1.30


def make_fig02(df):
    fig, axes = plt.subplots(1, 4, figsize=(22, 9), facecolor="white")
    # LAYOUT FIX: a touch more headroom at the top so the raised panel titles
    # clear both the suptitle and their own italic descriptor lines.
    fig.subplots_adjust(left=0.05, right=0.97, top=0.84, bottom=0.22, wspace=0.42)

    for ax, (metric, title, ylabel, subtitle, finding_note) in zip(axes, METRICS):
        positions, model_centres, offset = [], [], 0.0
        for model in MODEL_ORDER:
            cluster = []
            for cond in COND_ORDER:
                pos = offset + len(cluster) * WITHIN_GAP
                cluster.append(pos)
                positions.append((model, cond, pos))
            model_centres.append(float(np.mean(cluster)))
            offset = cluster[-1] + GROUP_GAP + WITHIN_GAP

        for model, cond, pos in positions:
            vals = df[(df["model"] == model) & (df["condition"] == cond)][metric].dropna().values
            col = COND_COLORS[cond]
            ax.boxplot(
                vals, positions=[pos], widths=BOX_W, patch_artist=True,
                notch=False, showfliers=True,
                whiskerprops=dict(color=col, lw=1.5),
                capprops=dict(color=col, lw=2.0),
                medianprops=dict(color="white", lw=2.5),
                flierprops=dict(marker="o", markerfacecolor=col, markersize=4,
                                alpha=0.50, markeredgewidth=0),
                boxprops=dict(facecolor=col, alpha=0.72, edgecolor=col, lw=1.0),
            )

        all_vals = df[metric].dropna().values
        bval = BASELINE_GRAND[metric]
        y_lo = min(np.percentile(all_vals, 1), bval)
        y_hi = max(np.percentile(all_vals, 99), bval)
        span = y_hi - y_lo
        ax.set_ylim(y_lo - span * 0.10, y_hi + span * 0.32)
        ax.axhline(bval, color="#E24B4A", lw=1.8, ls="--", alpha=0.88, zorder=0)

        x_positions = [p for _, _, p in positions]
        q75 = np.percentile(all_vals, 75)
        q25 = np.percentile(all_vals, 25)
        if BASELINE_ABOVE_DATA[metric]:
            label_y = max(bval + span * 0.04, q75 + span * 0.06)
        else:
            label_y = min(bval - span * 0.04, q25 - span * 0.06)
        ax.text(np.mean(x_positions), label_y, f"Statute ref: {bval:.2f}\n{finding_note}",
                ha="center", va="bottom", fontsize=7.0, color="#c0392b", linespacing=1.35,
                bbox=dict(facecolor="white", alpha=0.82, edgecolor="#f0a0a0",
                          boxstyle="round,pad=0.3", lw=0.6))

        ax_ylim = ax.get_ylim()
        y_span = ax_ylim[1] - ax_ylim[0]
        for model, cond, pos in positions:
            vals = df[(df["model"] == model) & (df["condition"] == cond)][metric].dropna()
            ax.text(pos, ax_ylim[0] + y_span * 0.01, f"n={len(vals)}",
                    ha="center", va="bottom", fontsize=6.5, color="#555")
        for mc, model in zip(model_centres, MODEL_ORDER):
            ax.text(mc, ax_ylim[0] - y_span * 0.09, model, ha="center", va="top",
                    fontsize=10, fontweight="bold", color="#2c2c2c")
        for model in MODEL_ORDER:
            cps = [pos for m, c, pos in positions if m == model]
            x0 = cps[0] - BOX_W / 2 - 0.06
            x1 = cps[-1] + BOX_W / 2 + 0.06
            y_b = ax_ylim[0] - y_span * 0.05
            ax.annotate("", xy=(x0, y_b), xytext=(x1, y_b), xycoords="data", textcoords="data",
                        arrowprops=dict(arrowstyle="-", color="#cccccc", lw=1.0),
                        annotation_clip=False)

        ax.set_xlim(min(x_positions) - BOX_W, max(x_positions) + BOX_W + 0.3)
        ax.set_xticks([])
        ax.set_ylabel(ylabel, fontsize=9)
        # LAYOUT FIX: raise the panel title (pad 6 -> 24) and drop the italic
        # descriptor just above the axes so the two no longer overlap.
        ax.set_title(title, fontsize=10, pad=24, fontweight="bold")
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.0, color="#666666", style="italic")
        ax.grid(axis="y", ls="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8.5)

    cond_handles = [mpatches.Patch(facecolor=COND_COLORS[c], alpha=0.78, label=c) for c in COND_ORDER]
    base_h = mlines.Line2D([], [], color="#E24B4A", ls="--", lw=1.8,
                           label="Baseline statute grand mean (avg. of 3 statutes)")
    fig.legend(handles=cond_handles + [base_h], title="Prompt condition", title_fontsize=9,
               loc="lower center", ncol=4, fontsize=9.5, framealpha=0.93,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.015))
    fig.suptitle(
        "Stage 3 — Complexity Compression: Structural Degradation Across Prompt Conditions\n"
        "Box = IQR  ·  Whiskers = 1.5×IQR  ·  White bar = median  ·  "
        "n = 60 per condition (30 per model)",
        fontsize=11.5, y=0.965, va="top",
    )

    out = os.path.join(FIG_DIR, "Fig02_Stage3_Complexity_Compression.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def make_fig03(df):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig = plt.figure(figsize=(10, 6))
    ax2 = sns.boxplot(
        data=df, x="condition", y="sent_var_red", hue="model", order=COND_ORDER,
        palette=["#1f77b4", "#ff7f0e"], linewidth=1.5, fliersize=4,
    )
    plt.title("The Collapse of Syntactic Variance Across AI Models\n"
              "(0% = each domain's baseline statute variance)",
              fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Prompt Constraint Level", fontsize=14, fontweight="bold")
    plt.ylabel("Sentence Length Variance Reduction", fontsize=14, fontweight="bold")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0, symbol="%"))
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Domain statute baseline (0% per domain)")
    # LAYOUT FIX: legend was loc="lower left", sitting on top of the boxplots.
    # The upper-right region is empty, so move it there.
    plt.legend(title="AI Model", title_fontsize="13", fontsize="12", loc="upper right",
               framealpha=0.95)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "Fig03_Stage3_Variance_Reduction.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    sns.reset_defaults()


def main():
    print("Regenerating figures with corrected layouts (data unchanged)...")
    make_fig01()
    df3 = pd.read_csv(STAGE3_CSV)
    print(f"  Fig02/Fig03: {len(df3)} drafts")
    make_fig02(df3)
    make_fig03(df3)
    print("Done.")


if __name__ == "__main__":
    main()
