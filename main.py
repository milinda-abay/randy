from utils.preprocess import preprocess_data
from analysis import distributions


from analysis import chi_squared, pairwise_cramer_heatmap


def main():
    data = preprocess_data()

    distributions.run(data)


    chi_squared.run(data, predictor='gender', target='password_storage')
    chi_squared.run(data, predictor='password_reuse', target='password_change')
    chi_squared.run(data, predictor='age_group', target='password_reuse')
    chi_squared.run(data, predictor='age_group', target='password_length')
    chi_squared.run(data, predictor='password_reuse', target='password_consistency')
    chi_squared.run(data, predictor='shared_passwords', target='password_sharing')
    pairwise_cramer_heatmap.run(data)


if __name__ == "__main__":
    main()
