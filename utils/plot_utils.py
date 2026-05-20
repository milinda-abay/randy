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
    n_x = counts.sum(axis=1)
    proportions = counts.div(n_x, axis=0)
    margin_pct = z * (proportions * (1 - proportions)).div(n_x, axis=0) ** 0.5 * 100

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
            yerr=margin_pct[level],
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


def plot_likert(data, col, group_col='age_group', palette='RdPu', figsize=(12, 6)):
    counts = data.groupby([group_col, col], observed=True).size().unstack(fill_value=0)
    cats = data[col].cat.categories.tolist()
    counts = counts.reindex(columns=cats, fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0) * 100

    n = len(cats)
    left_cats = cats[:n // 2]
    right_cats = cats[n // 2:]

    colors = sns.color_palette(palette, n_colors=n)
    cat_colors = dict(zip(cats, colors))

    groups = counts.index.tolist()
    y = list(range(len(groups)))

    fig, ax = plt.subplots(figsize=figsize)

    left_start = [0.0] * len(groups)
    for cat in reversed(left_cats):
        vals = props[cat].values
        ax.barh(y, -vals, left=[-s for s in left_start], color=cat_colors[cat])
        left_start = [s + v for s, v in zip(left_start, vals)]

    right_start = [0.0] * len(groups)
    for cat in right_cats:
        vals = props[cat].values
        ax.barh(y, vals, left=right_start, color=cat_colors[cat])
        right_start = [s + v for s, v in zip(right_start, vals)]

    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(x):.0f}%'))
    ax.set_xlabel('Percentage (%)')
    ax.set_ylabel(_to_title(group_col))
    ax.set_title(f'{_to_title(col)} by {_to_title(group_col)}')

    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_colors[c]) for c in cats]
    ax.legend(handles, cats, title=_to_title(col), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{col}_by_{group_col}_likert.png', bbox_inches='tight')
    plt.show()


def plot_heatmap(data, row_col, col_col, cmap='RdPu', figsize=(7, 6)):
    ct = pd.crosstab(data[row_col], data[col_col])

    if data[row_col].cat.ordered:
        ct = ct.loc[::-1]
    if data[col_col].cat.ordered:
        ct = ct.loc[:, ::-1]

    pct = ct / ct.values.sum() * 100
    annot = ct.astype(str) + '\n(' + pct.round(1).astype(str) + '%)'

    row_label = _to_title(row_col)
    col_label = _to_title(col_col)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pct, annot=annot, fmt='', cmap=cmap, ax=ax,
                cbar_kws={'label': 'Percentage (%)'})
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(f'{row_label} vs {col_label}')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{row_col}_vs_{col_col}.png', bbox_inches='tight')
    plt.show()

