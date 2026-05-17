import os
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
sns.set_style("whitegrid")

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def _to_title(col):
    return col.replace('_', ' ').title()


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
    plt.savefig(os.path.join(FIGURES_DIR, f'{bar_col}_by_{x_col}.png'), bbox_inches='tight')
    plt.show()
