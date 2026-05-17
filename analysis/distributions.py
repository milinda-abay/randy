from utils.preprocess import preprocess_data
from utils.plot_utils import plot_grouped_bar, plot_heatmap


def run(data):
    plot_grouped_bar(data, 'age_group', 'gender')
    plot_grouped_bar(data, 'age_group', 'password_reuse')
    plot_grouped_bar(data, 'age_group', 'password_length')
    plot_grouped_bar(data, 'age_group', 'password_inclusion')
    plot_heatmap(data, 'password_reuse', 'password_change')
