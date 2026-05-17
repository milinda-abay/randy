import pandas as pd
import pingouin as pg
from pathlib import Path
from utils.folders import DATA_DIR
from great_tables import GT




def run(data,predictor ,target):
    expected, observed, stats = pg.chi2_independence(data, x=predictor, y=target)
    expected = expected.round(3)
    stats = stats.rename(columns={'n': 'Total', 'dof': 'Degrees of Freedom', 'chi2': 'Chi-Squared', 'pval': 'P-Value', 'cramer': "Cramer's V"})
    stats =stats.round(3)

    gt_tbl = GT(stats)
    gt_tbl.save(DATA_DIR / f'{predictor}_vs_{target}_chi2_stats.png')
    print(gt_tbl)
    observed.to_csv(DATA_DIR / f'{predictor}_vs_{target}_observed.csv')
    expected.to_csv(DATA_DIR / f'{predictor}_vs_{target}_expected.csv')
    stats.to_csv(DATA_DIR / f'{predictor}_vs_{target}_stats.csv', index=False)
