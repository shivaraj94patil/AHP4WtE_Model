import pandas as pd
import numpy as np

def load_capex_data(filepath):
    """
    Load CAPEX data from CSV.
    Expects columns: Technology, Min_CAPEX, Max_CAPEX (EUR/t)
    """
    df = pd.read_csv(filepath)
    df["Avg_CAPEX"] = (df["Min_CAPEX"] + df["Max_CAPEX"]) / 2
    return df

def capex_to_pairwise(df):
    """
    Convert CAPEX data to a pairwise comparison matrix.
    Lower CAPEX = higher preference.
    """
    techs = df["Technology"].tolist()
    avg_costs = df["Avg_CAPEX"].to_numpy()
    size = len(techs)

    matrix = np.ones((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                ratio = avg_costs[j] / avg_costs[i]
                matrix[i, j] = saaty_scale(ratio)
    return techs, matrix

def saaty_scale(ratio):
    """
    Map a cost ratio to Saaty's 1–9 scale.
    Lower cost → higher preference.
    """
    if ratio >= 1:
        score = min(9, round(ratio))
    else:
        score = 1 / min(9, round(1 / ratio))
    return score
