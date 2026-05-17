from preprocess import preprocess_data
from plot_utils import plot_grouped_bar


def analyze_age_distribution(data):
    plot_grouped_bar(data, 'age_group', 'gender')


def analyze_password_reuse_by_age(data):
    plot_grouped_bar(data, 'age_group', 'password_reuse')


def analyze_password_length_by_age(data):
    plot_grouped_bar(data, 'age_group', 'password_length')


def analyze_password_inclusion_by_age(data):
    plot_grouped_bar(data, 'age_group', 'password_inclusion')


if __name__ == "__main__":
    processed_data = preprocess_data('survey_data.csv')
    analyze_age_distribution(processed_data)
    analyze_password_reuse_by_age(processed_data)
    analyze_password_length_by_age(processed_data)
    analyze_password_inclusion_by_age(processed_data)
