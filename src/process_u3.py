import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def process_feature_table(path_to_data, filename):
    """Read feature table and process it."""
    df = pd.read_csv(f"{path_to_data}/{filename}.tsv", sep="\t", index_col=0)
    print(f"Original shape {df.shape}")
    # extract only name of feature within "[]" and remove rest
    df.columns = df.columns.str.extract(r"\[([^]]+)\]", expand=False)
    # assuming "-1" is "unclassified"
    df.columns = df.columns.fillna("unclassified")
    return df


def preprocess_data_for_model(
    abundance_df: pd.DataFrame, cutoff: float = 1e-4, pseudocount: float = 1e-4
):
    """
    Filter low‐abundance species, log‐transform and scale features. According to
    original publication's R script in
    https://github.com/grp-bork/CellCount_Nishijima_2024/blob/main/construct_models.R
    """
    # todo: add shannon diversity
    df_with_div = abundance_df.copy()
    # df_with_div['shannon_diversity'] = shannon_vals

    # filter minor species
    mean_abund = df_with_div.mean(axis=0)
    prevalence = (df_with_div > 0).mean(axis=0)
    keep_mask = (mean_abund > cutoff) & (prevalence > 0.1)
    filtered_df = df_with_div.loc[:, keep_mask]
    feature_names = filtered_df.columns

    # log‐transform + scale
    log_df = np.log10(filtered_df + pseudocount)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(log_df)

    return features_scaled, feature_names, scaler
