from matplotlib import figure

from utils.preprocess import preprocess_data
from utils.plot_utils import plot_grouped_bar, plot_heatmap,plot_likert


def run(data):
    plot_grouped_bar(data, 'age_group', 'gender')
    plot_grouped_bar(data, 'age_group', 'password_reuse')
    plot_grouped_bar(data, 'age_group', 'password_length')
    plot_grouped_bar(data, 'age_group', 'password_inclusion')
    plot_grouped_bar(data, 'age_group', 'improvement')
    plot_heatmap(data, 'password_reuse', 'password_change')

    plot_grouped_bar(data, 'age_group', 'intention')
    plot_heatmap(data, 'age_group', 'intention', figsize=(9, 6))
    plot_grouped_bar(data, 'password_reuse', 'password_consistency')
    plot_heatmap(data, 'password_length', 'password_consistency')
    plot_grouped_bar(data, 'age_group', 'password_creation')
    plot_grouped_bar(data, 'password_inclusion', 'password_storage')
    plot_likert(data, col='password_creation', group_col='age_group', palette='RdPu', figsize=(12, 6))

    
if __name__ == "__main__":
    data = preprocess_data()
    run(data)