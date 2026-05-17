import pandas as pd
import pingouin as pg
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


def run(data):
    result = pg.chi2_independence(data, x='gender', y='password_storage')
    print(result)
    result[2].to_csv(DATA_DIR / 'gender_vs_storage_chi2.csv')
