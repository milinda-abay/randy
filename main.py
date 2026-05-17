from preprocess import preprocess_data

def main():
    processed_data = preprocess_data('survey_data.csv')
    print(processed_data.head())


if __name__ == "__main__":
    main()
