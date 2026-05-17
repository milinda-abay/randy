# function to analyze password length preferences by age group and visualize it using a bar chart with 95% confidence intervals.

import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import preprocess_data

sns.set_theme()
sns.set_style("whitegrid")
sns.set_palette(palette='RdYlGn', n_colors=4)

def analyze_password_length_by_age(data):
    counts = data.groupby(['age_group', 'password_length'], observed=True).size().unstack(fill_value=0)

    z = 1.96  # 95% CI
    n_age = counts.sum(axis=1)
    proportions = counts.div(n_age, axis=0)
    margin_pct = z * (proportions * (1 - proportions)).div(n_age, axis=0) ** 0.5 * 100

    age_groups = counts.index.astype(str).tolist()
    x = list(range(len(age_groups)))
    length_levels = counts.columns.tolist()
    width = 0.8 / len(length_levels)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, level in enumerate(length_levels):
        offset = (i - len(length_levels) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            proportions[level] * 100,
            width,
            label=level,
            yerr=margin_pct[level],
            capsize=5,
        )

    ax.set_xlabel('Age Group')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Password Length Preference by Age Group with 95% Confidence Intervals')
    ax.set_xticks(x)
    ax.set_xticklabels(age_groups, rotation=45)
    ax.legend(title='Password Length')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    processed_data = preprocess_data('survey_data.csv')
    analyze_password_length_by_age(processed_data)
