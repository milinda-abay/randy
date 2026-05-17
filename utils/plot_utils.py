import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
sns.set_style("whitegrid")

FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


def _to_title(col):
    return col.replace('_', ' ').title()


def _wilson_yerr(proportions_col, lower_col, upper_col):
    lo = (proportions_col * 100 - lower_col).clip(lower=0).values
    hi = (upper_col - proportions_col * 100).clip(lower=0).values
    return [lo, hi]


def plot_grouped_bar(data, x_col, bar_col, palette='RdPu', figsize=(14, 6)):
    counts = data.groupby([x_col, bar_col], observed=True).size().unstack(fill_value=0)

    z = 1.96
    z2 = z ** 2
    n_x = counts.sum(axis=1)
    proportions = counts.div(n_x, axis=0)

    denominator = 1 + z2 / n_x
    center = proportions.add(z2 / (2 * n_x), axis=0).div(denominator, axis=0)
    half_width = (
        z * (proportions * (1 - proportions)).div(n_x, axis=0)
        .add(z2 / (4 * n_x ** 2), axis=0)
        .pow(0.5)
        .div(denominator, axis=0)
    )
    lower = (center - half_width).clip(lower=0) * 100
    upper = (center + half_width).clip(upper=1) * 100

    bar_levels = counts.columns.astype(str).tolist()
    x_labels = counts.index.tolist()
    x = list(range(len(x_labels)))
    width = 0.8 / len(bar_levels)

    sns.set_palette(palette=palette, n_colors=len(bar_levels))

    fig, ax = plt.subplots(figsize=figsize)
    for i, level in enumerate(bar_levels):
        offset = (i - len(bar_levels) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            proportions[level] * 100,
            width,
            label=level,
            yerr=_wilson_yerr(proportions[level], lower[level], upper[level]),
            capsize=5,
        )

    x_label = _to_title(x_col)
    legend_title = _to_title(bar_col)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'{legend_title} by {x_label} with 95% Confidence Intervals')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.legend(title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{bar_col}_by_{x_col}.png', bbox_inches='tight')
    plt.show()


def plot_heatmap(data, row_col, col_col, cmap='RdPu', figsize=(7, 6)):
    ct = pd.crosstab(data[row_col], data[col_col])

    if data[row_col].cat.ordered:
        ct = ct.loc[::-1]
    if data[col_col].cat.ordered:
        ct = ct.loc[:, ::-1]

    row_label = _to_title(row_col)
    col_label = _to_title(col_col)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ct, annot=True, fmt='d', cmap=cmap, ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(f'{row_label} vs {col_label}')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{row_col}_vs_{col_col}.png', bbox_inches='tight')
    plt.show()
