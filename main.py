from utils.preprocess import preprocess_data
from analysis.distributions import (
    analyze_age_distribution,
    analyze_password_reuse_by_age,
    analyze_password_length_by_age,
    analyze_password_inclusion_by_age,
    analyze_password_reuse_by_password_change,
)
from analysis import chi_squared, mca, pairwise_cramer_heatmap


def main():
    data = preprocess_data()

    analyze_age_distribution(data)
    analyze_password_reuse_by_age(data)
    analyze_password_length_by_age(data)
    analyze_password_inclusion_by_age(data)
    analyze_password_reuse_by_password_change(data)

    chi_squared.run(data)
    mca.run(data)
    pairwise_cramer_heatmap.run(data)


if __name__ == "__main__":
    main()
