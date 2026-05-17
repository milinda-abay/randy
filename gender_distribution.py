# function to analyze age distribution of the survey respondents and visualize it using a bar chart with 95% confidence intervals by gender.

import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import preprocess_data

sns.set_theme()
sns.set_style("whitegrid")
sns.set_palette(palette='RdPu', n_colors=2)

def analyze_age_distribution(data):
    counts = data.groupby(['age_group', 'gender'], observed=True).size().unstack(fill_value=0)

    z = 1.96  # 95% CI
    n_age = counts.sum(axis=1)
    proportions = counts.div(n_age, axis=0)
    margin_pct = z * (proportions * (1 - proportions)).div(n_age, axis=0) ** 0.5 * 100

    genders = counts.columns.astype(str).tolist()
    age_groups = counts.index.tolist()
    x = list(range(len(age_groups)))
    width = 0.8 / len(genders)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, gender in enumerate(genders):
        offset = (i - len(genders) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            proportions[gender] * 100,
            width,
            label=gender,
            yerr=margin_pct[gender],
            capsize=5,
        )

    ax.set_xlabel('Age Group')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Gender Distribution by Age Group with 95% Confidence Intervals')
    ax.set_xticks(x)
    ax.set_xticklabels(age_groups, rotation=45)
    ax.legend(title='Gender', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    processed_data = preprocess_data('survey_data.csv')
    analyze_age_distribution(processed_data)
