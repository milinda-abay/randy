import pandas as pd

FILE_PATH = 'survey_data.csv'

rename_columns = {
'Response ID':"response_id",
'Please tick the most appropriate option:':"gender",
'2. What is your age group ?':"age_group",
'3. In your opinion, what do you think the length of a password SHOULD be?':"password_length",
'4. In your opinion, what do you think a password SHOULD usually include?':"password_inclusion",
'5. How often do you reuse the same passwords (e.g same password for multiple accounts) ?':"password_reuse",
'7. How often do you reuse the same password but change characters/numbers/symbols ?':"password_change",
'11. Where do you currently store your passwords?':"password_storage",
'12. Do you have any shared/common password with anyone? (such as friends, family or colleagues) ?':"shared_passwords",
'14. Have you ever shared/told anyone your password? (e.g friends, family, colleagues etc)':"password_sharing",
'16. How likely are you to use the same password for multiple accounts ?':"password_consistency",
'17. How do you create your passwords/s ?':"password_creation"
}

def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(FILE_PATH)

    # Drop these columns as they are not relevant to the analysis
    data = data.drop(columns=['Consent','Q6','Q8','Q9', 'Q10', 'Q13', 'Q15','Q18','Q19', 'Q20', 'Q21', 'Q22', 'Q23'])

    #convert the first row to column names
    data.columns = list(data.iloc[0])

    # Drop the first and second rows after setting the first row as column names
    data = data.drop([0, 1]).reset_index(drop=True)

    # Rename columns for better readability
    data = data.rename(columns=rename_columns)

    data = data.convert_dtypes()

    for col in data.columns:
        if col != 'response_id':
            data[col] = data[col].astype('category')

    data['gender'] = data['gender'].astype('category')

    return data

if __name__ == "__main__":
    processed_data = preprocess_data(FILE_PATH)
