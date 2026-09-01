"""Shared data loading and output helpers for the AutoML comparator arms.

Imported from several conda environments (`autosklearn`, `tpot_bench`,
`maml_bench`) whose scikit-learn versions differ by years, so this module
imports nothing beyond pandas and numpy.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def _read_wide_parquet(path: str) -> pd.DataFrame:
    """Read a parquet file whose schema exceeds pyarrow's default thrift limits.

    A 3x10^5-column table (u4) overflows the default metadata size caps at
    read time; raising them is the documented remedy and a no-op for narrow
    tables.
    """
    import pyarrow.parquet as pq

    table = pq.ParquetFile(
        path,
        thrift_string_size_limit=1_000_000_000,
        thrift_container_size_limit=1_000_000_000,
    ).read()
    return table.to_pandas()


def read_split(splits_dir: str, name: str) -> pd.DataFrame:
    """Read a ``train_val``/``test`` split frame, preferring parquet.

    Pickles embed NumPy module paths and cannot cross the NumPy 1<->2
    boundary; parquet round-trips cleanly.
    """
    parquet = os.path.join(splits_dir, f"{name}.parquet")
    if os.path.exists(parquet):
        return _read_wide_parquet(parquet)
    try:
        return pd.read_pickle(os.path.join(splits_dir, f"{name}.pkl"))
    except (ModuleNotFoundError, ImportError) as e:
        raise RuntimeError(
            f"{name}.pkl in {splits_dir} was written under a different NumPy "
            f"major version and no {name}.parquet exists. Run "
            f"`python -m src.launch_comparators --ensure-parquet` from the "
            f"ritme_usecases env first."
        ) from e


def _is_categorical(col: pd.Series) -> bool:
    return isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object


def enrich_categories(md_df: pd.DataFrame, enrich_with: list) -> dict:
    """Category universe per categorical enrichment feature, over the full table.

    Computed once and reused for every split so the dummy column set is
    identical across train and test.
    """
    return {
        feat: sorted(md_df[feat].dropna().unique())
        for feat in enrich_with
        if _is_categorical(md_df[feat])
    }


def add_enrichment(
    X: pd.DataFrame, md_df: pd.DataFrame, enrich_with: list, categories: dict
) -> pd.DataFrame:
    """Append ritme's ``data_enrich_with`` metadata features to ``X``.

    Mirrors ritme's `feature_space/enrich_features.py`: categorical columns
    become drop-first dummies over ``categories``; everything else (including
    bool) is cast to float and appended unchanged.
    """
    out = X.copy()
    for feat in enrich_with:
        col = md_df.loc[X.index, feat]
        if _is_categorical(col):
            col = col.astype(pd.CategoricalDtype(categories=categories[feat]))
            dummies = pd.get_dummies(col, prefix=feat, drop_first=True, dtype=float)
            out = pd.concat([out, dummies], axis=1)
        else:
            out[feat] = col.astype(float)
    return out


def load_xy(
    path_ft: str,
    path_md: str,
    splits_dir: str,
    target: str,
    task: str,
    group_by_column: Optional[str] = None,
    enrich_with: Optional[list] = None,
) -> tuple:
    """Build the train/test matrices every comparator arm consumes.

    Returns ``(X_train, y_train, X_test, y_test, groups)``. ``groups`` is the
    training-set values of ``group_by_column`` when the use case defines one,
    else None.
    """
    train_df = read_split(splits_dir, "train_val")
    test_df = read_split(splits_dir, "test")

    md_df = pd.read_csv(path_md, sep="\t", index_col=0)
    if str(path_ft).endswith(".biom"):
        # u4: the pre-staged split frames already hold the features as
        # F-prefixed relative abundances; the BIOM is never read here.
        feature_cols = [c for c in train_df.columns if c.startswith("F")]
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
    else:
        otu_df = pd.read_csv(path_ft, sep="\t", index_col=0)
        otu_df = otu_df.div(otu_df.sum(axis=1), axis=0)
        X_train = otu_df.loc[train_df.index]
        X_test = otu_df.loc[test_df.index]

    if enrich_with:
        missing = [f for f in enrich_with if f not in md_df.columns]
        if missing:
            raise KeyError(f"Enrichment features not in {path_md}: {missing}")
        categories = enrich_categories(md_df, enrich_with)
        X_train = add_enrichment(X_train, md_df, enrich_with, categories)
        X_test = add_enrichment(X_test, md_df, enrich_with, categories)

    y_train = md_df.loc[train_df.index, target]
    y_test = md_df.loc[test_df.index, target]
    if task == "classification":
        y_train, y_test = encode_labels(y_train, y_test)

    groups = None
    if group_by_column:
        groups = md_df.loc[train_df.index, group_by_column].to_numpy()

    return X_train, y_train, X_test, y_test, groups


def encode_labels(y_train: pd.Series, y_test: pd.Series) -> tuple:
    """Return integer class labels for both splits under one shared mapping.

    Numeric targets pass through as int. String targets (u4's ``empo_3``) are
    mapped to 0..k-1 in sorted order — several estimators in the comparator
    spaces (XGBoost among them) refuse string labels, and every reported
    metric is label-agnostic.
    """
    try:
        return y_train.astype(int), y_test.astype(int)
    except (ValueError, TypeError):
        classes = sorted(pd.concat([y_train, y_test]).dropna().unique())
        mapping = {c: i for i, c in enumerate(classes)}
        print(f"Encoded {len(classes)} string classes: {mapping}")
        return y_train.map(mapping).astype(int), y_test.map(mapping).astype(int)


def write_metrics(
    out_dir: str, usecase: str, method: str, metrics: pd.DataFrame
) -> str:
    """Write a metrics frame as ``<usecase>_<method>_metrics.csv``.

    The shared evaluators in `src.eval_automl` label every row ``automl``;
    relabel it with the arm's own name so the merged comparison table
    distinguishes them.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{usecase}_{method}_metrics.csv")
    out = metrics.copy()
    out.index = [method] * len(out)
    out = out.reset_index(names="model")
    out.to_csv(path, index=False)
    return path


def write_configs(
    out_dir: str, usecase: str, method: str, configs: pd.DataFrame
) -> str:
    """Write the evaluated-configuration table as ``<usecase>_<method>_configs.csv``."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{usecase}_{method}_configs.csv")
    configs.to_csv(path, index=False)
    return path
