from utils.preprocess import preprocess_data
from utils.plot_utils import plot_grouped_bar, plot_heatmap


def analyze_age_distribution(data):
    plot_grouped_bar(data, 'age_group', 'gender')


def analyze_password_reuse_by_age(data):
    plot_grouped_bar(data, 'age_group', 'password_reuse')


def analyze_password_length_by_age(data):
    plot_grouped_bar(data, 'age_group', 'password_length')


def analyze_password_inclusion_by_age(data):
    plot_grouped_bar(data, 'age_group', 'password_inclusion')


def analyze_password_reuse_by_password_change(data):
    plot_heatmap(data, 'password_reuse', 'password_change')
