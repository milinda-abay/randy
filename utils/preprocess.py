import pandas as pd
from pathlib import Path

from utils.folders import DATA_DIR

rename_columns = {
    'Response ID': "response_id",
    'Please tick the most appropriate option:': "gender",
    '2. What is your age group ?': "age_group",
    '3. In your opinion, what do you think the length of a password SHOULD be?': "password_length",
    '4. In your opinion, what do you think a password SHOULD usually include?': "password_inclusion",
    '5. How often do you reuse the same passwords (e.g same password for multiple accounts) ?': "password_reuse",
    '7. How often do you reuse the same password but change characters/numbers/symbols ?': "password_change",
    '11. Where do you currently store your passwords?': "password_storage",
    '12. Do you have any shared/common password with anyone? (such as friends, family or colleagues) ?': "shared_passwords",
    '14. Have you ever shared/told anyone your password? (e.g friends, family, colleagues etc)': "password_sharing",
    '16. How likely are you to use the same password for multiple accounts ?': "password_consistency",
    '17. How do you create your passwords/s ?': "password_creation"
}


def preprocess_data(file_path=None):
    if file_path is None:
        file_path = DATA_DIR / 'datafile_classification.csv'

    data = pd.read_csv(file_path)
    data = data.drop(columns=['Consent', 'Q6', 'Q8', 'Q9', 'Q10', 'Q13', 'Q15', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22', 'Q23'])
    data.columns = list(data.iloc[0])
    data = data.drop([0, 1]).reset_index(drop=True)
    data = data.rename(columns=rename_columns)

    cols = list(data.columns)
    cols[12] = "improvement"
    cols[13] = "intention"
    data.columns = cols

    data = data.convert_dtypes()

    for col in ["password_inclusion", "password_change", "password_storage", "password_creation", "password_consistency"]:
        data[col] = data[col].str.capitalize()
    data['password_consistency'] = data['password_consistency'].str.replace('very unlikely', 'Very unlikely', regex=False)

    
    data = data.dropna(how='all',subset = data.columns.difference(['response_id'])).reset_index(drop=True)

    category_dtypes = {
        'gender': pd.CategoricalDtype(categories=['Female', 'Male'], ordered=False),
        'age_group': pd.CategoricalDtype(categories=['18-24', '25-34', '35-44', '45-54', '55-64', 'over 65'], ordered=True),
        'password_length': pd.CategoricalDtype(categories=['Less than 8 characters', '8-12 characters', '12-16 characters', 'More that 16 characters'], ordered=True),
        'password_inclusion': pd.CategoricalDtype(categories=['Only symbols', 'Only letters', 'Letters and numbers', 'Letters, numbers and symbols', 'A passphrase'], ordered=True),
        'password_reuse': pd.CategoricalDtype(categories=['Never', 'Sometimes', 'Often', 'Always'], ordered=True),
        'password_change': pd.CategoricalDtype(categories=['Never', 'Sometimes', 'Often', 'Always'], ordered=True),
        'password_storage': pd.CategoricalDtype(categories=["I don't store my password anywhere", 'On a sticky note', 'In a book/diary', 'On my phone notes app', 'On a document', 'On password manager'], ordered=True),
        'shared_passwords': pd.CategoricalDtype(categories=['Yes', 'No'], ordered=False),
        'password_sharing': pd.CategoricalDtype(categories=['Yes', 'No'], ordered=False),
        'password_consistency': pd.CategoricalDtype(categories=['Very unlikely', 'Somewhat unlikely', 'Somewhat likely', 'Very likely'], ordered=True),
        'password_creation': pd.CategoricalDtype(categories=['I create my own password', 'I use password generation tool', 'Other'], ordered=False),
        'improvement': pd.CategoricalDtype(categories=['No improvement', 'Moderate improvement', 'Substantial improvement'], ordered=False),
        'intention': pd.CategoricalDtype(categories=['No will not change', 'No change', 'Maybe - Might change', 'Yes will change'], ordered=True),
    }

    

    for col, dtype in category_dtypes.items():
        data[col] = data[col].astype(dtype)

    return data


if __name__ == "__main__":
    processed_data = preprocess_data()
    processed_data.to_csv(DATA_DIR / 'processed_datafile_classification.csv', index=False)
