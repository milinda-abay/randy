"""
Multiple Correspondence Analysis (MCA) — password survey data
=============================================================
Produces four publication-ready plots:
  1. Scree plot          — how many dimensions to retain
  2. Biplot              — categories + respondents in MCA space
  3. Category biplot     — categories only, sized by cos² quality
  4. Variable map        — per-variable mean coordinates (clean overview)

Dependencies:
    pip install prince pandas numpy matplotlib scipy adjustText
"""

import itertools
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import prince
from scipy.stats import chi2_contingency

try:
    from adjustText import adjust_text
    HAS_ADJUST = True
except ImportError:
    HAS_ADJUST = False

FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

VAR_COLOURS = {
    "gender":               "#378ADD",
    "age_group":            "#D85A30",
    "password_length":      "#1D9E75",
    "password_inclusion":   "#7F77DD",
    "password_reuse":       "#D4537E",
    "password_change":      "#BA7517",
    "password_storage":     "#888780",
    "shared_passwords":     "#639922",
    "password_sharing":     "#993C1D",
    "password_consistency": "#0F6E56",
    "password_creation":    "#534AB7",
}

SHORT = {
    "gender":               "Gender",
    "age_group":            "Age group",
    "password_length":      "Length",
    "password_inclusion":   "Inclusion",
    "password_reuse":       "Reuse",
    "password_change":      "Change freq.",
    "password_storage":     "Storage",
    "shared_passwords":     "Has shared",
    "password_sharing":     "Shares now",
    "password_consistency": "Consistency",
    "password_creation":    "Creation",
}

STYLE = dict(dpi=150, facecolor="white")


def var_of(label: str) -> str:
    return label.split("__")[0]


def cat_of(label: str) -> str:
    return label.split("__", 1)[1]


def run(data):
    cols = [c for c in data.columns if c != "response_id"]
    df = data[cols].dropna().reset_index(drop=True)

    print(f"Rows after dropping NAs: {len(df)}  (from {len(data)})")
    print(f"Variables: {len(cols)}")
    print(f"Categories: {sum(df[c].nunique() for c in cols)}\n")

    N_COMPONENTS = 5

    mca_model = prince.MCA(
        n_components=N_COMPONENTS,
        n_iter=10,
        random_state=42,
        one_hot=True,
        one_hot_prefix_sep="__",
    )
    mca_model.fit(df)

    row_coords = mca_model.row_coordinates(df)
    col_coords = mca_model.column_coordinates(df)
    cos2       = mca_model.column_cosine_similarities(df)

    pct_var = np.array(mca_model.percentage_of_variance_)
    cum_var = np.cumsum(pct_var)

    print("Variance explained per dimension:")
    for i, (p, c) in enumerate(zip(pct_var, cum_var)):
        print(f"  Dim {i+1}: {p:.2f}%   cumulative: {c:.2f}%")

    cos2_2d = cos2[[0, 1]].sum(axis=1)

    # ── Plot 1: Scree plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4), **STYLE)

    dims = np.arange(1, N_COMPONENTS + 1)
    bar_col = "#B5D4F4"
    line_col = "#185FA5"

    ax.bar(dims, pct_var, color=bar_col, edgecolor="#85B7EB", linewidth=0.6, zorder=2)
    ax_r = ax.twinx()
    ax_r.plot(dims, cum_var, color=line_col, marker="o", markersize=5,
              linewidth=1.5, label="Cumulative %", zorder=3)
    ax_r.set_ylim(0, 105)
    ax_r.set_ylabel("Cumulative variance (%)", fontsize=9, color=line_col)
    ax_r.tick_params(axis="y", labelcolor=line_col, labelsize=8)

    for i, (p, c) in enumerate(zip(pct_var, cum_var)):
        ax.text(i + 1, p + 0.2, f"{p:.1f}%", ha="center", va="bottom",
                fontsize=8, color="#0C447C")

    ax.set_xlabel("Dimension", fontsize=9)
    ax.set_ylabel("% variance explained", fontsize=9)
    ax.set_xticks(dims)
    ax.set_xticklabels([f"Dim {d}" for d in dims], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title("MCA scree plot — variance by dimension", fontsize=10, pad=10, loc="left")
    ax.grid(axis="y", linewidth=0.4, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mca_1_scree.png", **STYLE, bbox_inches="tight")
    plt.close()
    print("\nSaved → mca_1_scree.png")

    # ── Plot 2: Full biplot (rows + columns) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9), **STYLE)

    ax.scatter(
        row_coords[0], row_coords[1],
        c="#B4B2A9", s=18, alpha=0.45, linewidths=0,
        label="Respondent", zorder=2,
    )

    texts = []
    for label in col_coords.index:
        var  = var_of(label)
        cat  = cat_of(label)
        x, y = col_coords.loc[label, 0], col_coords.loc[label, 1]
        q    = cos2_2d[label]
        size = 40 + 220 * q

        ax.scatter(x, y, c=VAR_COLOURS[var], s=size, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=4)

        short_cat = textwrap.shorten(cat.replace("_", " "), width=22, placeholder="…")
        texts.append(ax.text(x, y, short_cat, fontsize=7.5,
                              color=VAR_COLOURS[var], zorder=5))

    if HAS_ADJUST:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5))

    ax.axhline(0, color="#cccccc", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#cccccc", linewidth=0.6, zorder=1)
    ax.set_xlabel(f"Dimension 1  ({pct_var[0]:.1f}% variance)", fontsize=9)
    ax.set_ylabel(f"Dimension 2  ({pct_var[1]:.1f}% variance)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title("MCA biplot — respondents and category coordinates\n"
                 "Point size ∝ cos² quality of representation in 2D",
                 fontsize=10, pad=10, loc="left")

    var_patches = [
        mpatches.Patch(facecolor=c, label=SHORT[v], edgecolor="white", linewidth=0.5)
        for v, c in VAR_COLOURS.items()
    ]
    respondent_handle = mlines.Line2D([], [], marker="o", color="#B4B2A9",
                                      markersize=5, linewidth=0, label="Respondent", alpha=0.6)
    ax.legend(
        handles=var_patches + [respondent_handle],
        loc="upper right", bbox_to_anchor=(1.18, 1.0),
        fontsize=7.5, frameon=True, framealpha=0.9,
        edgecolor="#cccccc", title="Variable", title_fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mca_2_biplot.png", **STYLE, bbox_inches="tight")
    plt.close()
    print("Saved → mca_2_biplot.png")

    # ── Plot 3: Category-only plot ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9), **STYLE)

    texts = []
    for label in col_coords.index:
        var  = var_of(label)
        cat  = cat_of(label)
        x, y = col_coords.loc[label, 0], col_coords.loc[label, 1]
        q    = cos2_2d[label]
        size = 55 + 280 * q
        alpha = 0.4 + 0.55 * q

        ax.scatter(x, y, c=VAR_COLOURS[var], s=size,
                   alpha=alpha, edgecolors="white", linewidths=0.6, zorder=3)

        short_cat = textwrap.shorten(cat.replace("_", " "), width=24, placeholder="…")
        txt = ax.text(x, y, short_cat, fontsize=8, color=VAR_COLOURS[var],
                      fontweight="bold" if q > 0.35 else "normal", zorder=4)
        texts.append(txt)

    if HAS_ADJUST:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.5))

    ax.axhline(0, color="#cccccc", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#cccccc", linewidth=0.6, zorder=1)
    ax.set_xlabel(f"Dimension 1  ({pct_var[0]:.1f}% variance)", fontsize=9)
    ax.set_ylabel(f"Dimension 2  ({pct_var[1]:.1f}% variance)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title("MCA category map — password survey\n"
                 "Size and opacity ∝ cos² (quality of 2D representation); bold = cos² > 0.35",
                 fontsize=10, pad=10, loc="left")

    var_patches = [
        mpatches.Patch(facecolor=c, label=SHORT[v], edgecolor="white", linewidth=0.5)
        for v, c in VAR_COLOURS.items()
    ]
    ax.legend(handles=var_patches, loc="upper right", bbox_to_anchor=(1.18, 1.0),
              fontsize=7.5, frameon=True, framealpha=0.9,
              edgecolor="#cccccc", title="Variable", title_fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mca_3_category_map.png", **STYLE, bbox_inches="tight")
    plt.close()
    print("Saved → mca_3_category_map.png")

    # ── Plot 4: Variable map ──────────────────────────────────────────────────
    var_centroids = {}
    for var in VAR_COLOURS:
        mask  = [l for l in col_coords.index if var_of(l) == var]
        w     = cos2_2d[mask].values
        w     = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
        cx    = (col_coords.loc[mask, 0].values * w).sum()
        cy    = (col_coords.loc[mask, 1].values * w).sum()
        var_centroids[var] = (cx, cy)

    fig, ax = plt.subplots(figsize=(9, 7), **STYLE)

    texts = []
    for var, (cx, cy) in var_centroids.items():
        ax.scatter(cx, cy, c=VAR_COLOURS[var], s=160,
                   edgecolors="white", linewidths=1.0, zorder=3)
        texts.append(ax.text(cx, cy + 0.03, SHORT[var], fontsize=9,
                              color=VAR_COLOURS[var], fontweight="500", zorder=4,
                              ha="center", va="bottom"))

    if HAS_ADJUST:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5))

    ax.axhline(0, color="#cccccc", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#cccccc", linewidth=0.6, zorder=1)
    ax.set_xlabel(f"Dimension 1  ({pct_var[0]:.1f}% variance)", fontsize=9)
    ax.set_ylabel(f"Dimension 2  ({pct_var[1]:.1f}% variance)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title("MCA variable map — cos²-weighted variable centroids\n"
                 "Variables close together share respondents with similar profiles",
                 fontsize=10, pad=10, loc="left")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mca_4_variable_map.png", **STYLE, bbox_inches="tight")
    plt.close()
    print("Saved → mca_4_variable_map.png")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n── Category coordinates (Dim 1 & 2) with cos² quality ──\n")
    summary = pd.DataFrame({
        "variable":  [var_of(l) for l in col_coords.index],
        "category":  [cat_of(l) for l in col_coords.index],
        "dim1":       col_coords[0].round(3),
        "dim2":       col_coords[1].round(3),
        "cos2_2d":    cos2_2d.round(3),
    }).sort_values("cos2_2d", ascending=False)

    print(summary.to_string(index=False))
    print("\nDone. Four plots saved to the figures directory.")
