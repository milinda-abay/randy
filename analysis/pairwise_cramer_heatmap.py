"""
Pairwise Cramér's V heatmap — password survey data
====================================================
Computes Cramér's V for every pair of categorical columns and renders
a annotated, triangular heatmap ordered by average association strength.

Dependencies: pandas, scipy, numpy, matplotlib, seaborn
    pip install pandas scipy numpy matplotlib seaborn
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    "gender":               "Gender",
    "age_group":            "Age group",
    "password_length":      "Pwd length",
    "password_inclusion":   "Pwd complexity",
    "password_reuse":       "Pwd reuse",
    "password_change":      "Pwd change",
    "password_storage":     "Pwd storage",
    "shared_passwords":     "Has shared",
    "password_sharing":     "Shares now",
    "password_consistency": "Consistency",
    "password_creation":    "Creation method",
}


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Bias-corrected Cramér's V for two categorical Series.

    Uses the correction proposed by Bergsma & Wicher (2013) which adjusts
    for the positive bias in the standard formula when k or r is large
    relative to n.

    Returns a value in [0, 1]:
        0 = complete independence
        1 = perfect association
    """
    ct = pd.crosstab(x, y)
    n = ct.values.sum()
    if n == 0:
        return 0.0

    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    r, k = ct.shape

    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))

    k_corr = k - (k - 1) ** 2 / (n - 1)
    r_corr = r - (r - 1) ** 2 / (n - 1)

    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0

    return float(np.sqrt(phi2_corr / denom))


def run(data):
    cols = [c for c in data.columns if c != "response_id"]

    print("Computing Cramér's V for all variable pairs …")

    v_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for c1, c2 in itertools.product(cols, repeat=2):
        if c1 == c2:
            v_matrix.loc[c1, c2] = 1.0
        else:
            subset = data[[c1, c2]].dropna()
            v_matrix.loc[c1, c2] = cramers_v(subset[c1], subset[c2])

    print("Done.\n")
    print("Top 10 strongest pairs:")
    pairs = [
        (c1, c2, v_matrix.loc[c1, c2])
        for c1, c2 in itertools.combinations(cols, 2)
    ]
    for c1, c2, v in sorted(pairs, key=lambda x: -x[2])[:10]:
        print(f"  {LABELS[c1]:20s}  ×  {LABELS[c2]:20s}  =  {v:.3f}")

    mean_v = (v_matrix.sum(axis=1) - 1.0) / (len(cols) - 1)
    order  = mean_v.sort_values(ascending=False).index.tolist()

    v_ordered = v_matrix.loc[order, order].astype(float)

    mask = np.triu(np.ones(v_ordered.shape, dtype=bool), k=1)

    def v_label_colour(v: float) -> str:
        return "white" if v > 0.55 else "black"

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("white")

    cmap = sns.light_palette("#1D9E75", as_cmap=True)

    sns.heatmap(
        v_ordered,
        mask=mask,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="#e8e8e8",
        annot_kws={"size": 9, "weight": "normal"},
        square=True,
        cbar_kws={"shrink": 0.6, "label": "Cramér's V"},
        ax=ax,
    )

    for text in ax.texts:
        try:
            v = float(text.get_text())
            text.set_color(v_label_colour(v))
        except ValueError:
            pass

    tick_labels = [LABELS.get(c, c) for c in order]
    ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=9)
    ax.tick_params(length=0)

    ax.set_title(
        "Pairwise Cramér's V — password survey\n"
        "Columns ordered by mean association strength (high → low)",
        fontsize=11,
        pad=14,
        loc="left",
    )

    legend_items = [
        mpatches.Patch(facecolor=sns.light_palette("#1D9E75", n_colors=9)[2], label="Weak  (V < 0.20)"),
        mpatches.Patch(facecolor=sns.light_palette("#1D9E75", n_colors=9)[5], label="Moderate  (0.20 – 0.40)"),
        mpatches.Patch(facecolor=sns.light_palette("#1D9E75", n_colors=9)[8], label="Strong  (V > 0.40)"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.18),
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        title="Association strength",
        title_fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cramers_v_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved → cramers_v_heatmap.png")
