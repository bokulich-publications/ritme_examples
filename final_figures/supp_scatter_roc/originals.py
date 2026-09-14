"""Refit the published baseline of each use case and return its predictions.

Each recipe is the one in that use case's ``n4_original_setup.ipynb``: same
feature table, same split, same hyperparameters, same seeds. u4 additionally
reproduces EMP's caret pipeline (``nearZeroVar`` + ``findCorrelation`` +
``mtry`` tuning by Kappa under the oneSE rule), whose tuning results are read
from the cache the notebook wrote.

Runs in the ``ritme_usecases`` environment.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler

from src.eval_originals import load_baseline_split

try:
    from final_figures.supp_scatter_roc.arms import REPO_ROOT
except ImportError:  # running from inside final_figures/supp_scatter_roc/
    from arms import REPO_ROOT  # type: ignore[no-redef]

DATA = REPO_ROOT / "data"
U4_SPLITS = (
    REPO_ROOT
    / "use_cases"
    / "u4_amplicon_emp_classification"
    / "data_splits_u4_rare5000"
)
U4_CACHE = REPO_ROOT / "use_cases" / "result_figures" / "_cache_u4_original"
U4_METADATA_COLUMNS = ["empo_3", "empo_2", "study_id", "split"]
U4_TARGET = "empo_3"
U4_SEED = 12
U4_N_ESTIMATORS = 500
U4_CORR_CUTOFF = 0.9
U4_PREDICT_CHUNK = 2000


def _splits(usecase: str, ft: str, md: str, target: str):
    splits_dir = {
        "u1": "u1_amplicon_age_prediction/data_splits_u1",
        "u2": "u2_metagenome_ocean/data_splits_u2",
        "u3": "u3_amplicon_crc_classification/data_splits_u3",
    }[usecase]
    return load_baseline_split(
        str(REPO_ROOT / "use_cases" / splits_dir),
        str(DATA / ft),
        str(DATA / md),
        target,
    )


def u1():
    """Subramanian 2014: random forest on relative abundances of the rarefied table."""
    X_train, y_train, X_test, y_test = _splits(
        "u1",
        "u1_subramanian14/otu_table_subr14_rar.tsv",
        "u1_subramanian14/md_subr14_rar.tsv",
        "age_months",
    )
    X_train = X_train.div(X_train.sum(axis=1), axis=0)
    X_test = X_test.div(X_test.sum(axis=1), axis=0)
    model = RandomForestRegressor(
        n_estimators=10000,
        max_features=round(X_train.shape[1] / 3),
        random_state=123,
        n_jobs=-1,
    ).fit(X_train, y_train)
    return {
        "train": (y_train, model.predict(X_train)),
        "test": (y_test, model.predict(X_test)),
    }


def u2():
    """Sunagawa 2015: cross-validated elastic net on the processed OTU table."""
    X_train, y_train, X_test, y_test = _splits(
        "u2",
        "u2_tara_ocean/otu_table_tara_ocean_proc.tsv",
        "u2_tara_ocean/md_tara_ocean.tsv",
        "temperature_mean_degc",
    )
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, random_state=123, n_jobs=-1
    ).fit(X_train, y_train)
    return {
        "train": (y_train, model.predict(X_train)),
        "test": (y_test, model.predict(X_test)),
    }


def u3():
    """Topcuoglu 2020: min-max scaling + grid-searched RF on the subsampled table."""
    X_train, y_train, X_test, y_test = _splits(
        "u3",
        "u3_topcuoglu20_baxter/otu_table_baxter_subsampled.tsv",
        "u3_topcuoglu20_baxter/md_baxter.tsv",
        "srn",
    )
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=X_test.columns
    ).clip(lower=0, upper=1)

    search = GridSearchCV(
        RandomForestClassifier(n_estimators=500, random_state=12, n_jobs=-1),
        {"max_features": ["sqrt", "log2", 0.05, 0.1], "min_samples_leaf": [1, 5]},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=12),
        scoring="roc_auc",
        n_jobs=-1,
    ).fit(X_train, y_train)
    model = search.best_estimator_
    print(f"best CV AUROC {search.best_score_:.4f} at {search.best_params_}")
    return {
        "train": (y_train, model.predict_proba(X_train), list(model.classes_)),
        "test": (y_test, model.predict_proba(X_test), list(model.classes_)),
    }


def _load_features(frame, dtype=np.float32):
    head = list(frame.columns[: len(U4_METADATA_COLUMNS)])
    if head != U4_METADATA_COLUMNS:
        raise ValueError(f"unexpected metadata columns: {head}")
    return frame.iloc[:, len(U4_METADATA_COLUMNS) :].to_numpy(dtype=dtype)


def _near_zero_var(X, freq_cut=99.0, unique_cut=1.0, chunk=2000):
    """``caret::nearZeroVar``: boolean mask of columns to drop."""
    n, p = X.shape
    drop = np.zeros(p, dtype=bool)
    for start in range(0, p, chunk):
        block = np.asarray(X[:, start : start + chunk])
        for j in range(block.shape[1]):
            counts = np.unique(block[:, j], return_counts=True)[1]
            if counts.size == 1:
                drop[start + j] = True
                continue
            counts = np.sort(counts)[::-1]
            drop[start + j] = (
                counts[0] / counts[1] > freq_cut
                and 100.0 * counts.size / n <= unique_cut
            )
    return drop


def _find_correlation(corr, cutoff=U4_CORR_CUTOFF):
    """``caret::findCorrelation`` with ``exact = FALSE``: indices to drop."""
    corr = np.asarray(corr)
    rank = rankdata(np.abs(corr).mean(axis=0), method="dense")
    mask = np.abs(corr) > cutoff
    np.fill_diagonal(mask, False)
    rows, cols = np.nonzero(mask)
    upper = rows < cols
    rows, cols = rows[upper], cols[upper]
    discard_col = rank[cols] > rank[rows]
    return np.unique(np.concatenate([cols[discard_col], rows[~discard_col]]))


def _one_se_choice(mean_metric, sd_metric, n_resamples):
    """``caret``'s ``oneSE``: the smallest ``mtry`` within one SE of the best."""
    best = int(np.argmax(mean_metric))
    threshold = mean_metric[best] - sd_metric[best] / np.sqrt(n_resamples)
    return int(min(i for i, m in enumerate(mean_metric) if m >= threshold))


def _u4_tune(X, y):
    """Kappa per ``mtry`` under 10-fold CV repeated 5 times, resumed from cache."""
    grid = [int(v) for v in np.floor(np.linspace(10, X.shape[1], 10))]
    folds = list(
        RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=U4_SEED).split(
            X, y
        )
    )
    os.makedirs(U4_CACHE, exist_ok=True)
    scores = {}
    for mtry in grid:
        cache = U4_CACHE / f"rare5000_mtry{mtry}.json"
        if cache.exists():
            scores[mtry] = json.loads(cache.read_text())
            continue
        kappas = [
            float(
                cohen_kappa_score(
                    y[val],
                    RandomForestClassifier(
                        n_estimators=U4_N_ESTIMATORS,
                        max_features=mtry,
                        random_state=U4_SEED,
                        n_jobs=-1,
                    )
                    .fit(X[train], y[train])
                    .predict(X[val]),
                )
            )
            for train, val in folds
        ]
        cache.write_text(json.dumps(kappas))
        scores[mtry] = kappas
        print(f"  mtry={mtry}: Kappa {np.mean(kappas):.4f}", flush=True)
    means = [float(np.mean(scores[m])) for m in grid]
    sds = [float(np.std(scores[m], ddof=1)) for m in grid]
    return grid, means, sds, len(folds)


def _u4_predict_proba(model, frame, mask):
    out = np.empty((len(frame), len(model.classes_)), dtype=np.float64)
    for start in range(0, len(frame), U4_PREDICT_CHUNK):
        stop = min(start + U4_PREDICT_CHUNK, len(frame))
        block = _load_features(frame.iloc[start:stop])
        out[start:stop] = model.predict_proba(np.ascontiguousarray(block[:, mask]))
    return out


def u4():
    """Thompson 2017: EMP's caret random-forest pipeline on the rarefied split.

    Splits are loaded and released one at a time: the rarefied test frame alone
    is tens of gigabytes in memory.
    """
    train = pd.read_pickle(U4_SPLITS / "train_val.pkl")
    X_train = _load_features(train)
    y_train = train[U4_TARGET].to_numpy()

    X64 = np.asarray(X_train, dtype=np.float64)
    nzv = _near_zero_var(X64)
    kept = np.flatnonzero(~nzv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(X64[:, kept], rowvar=False)
    del X64
    mask = np.zeros(X_train.shape[1], dtype=bool)
    mask[np.delete(kept, _find_correlation(corr))] = True
    del corr
    X_train = np.ascontiguousarray(X_train[:, mask])
    print(f"  caret filters kept {mask.sum()} of {len(mask)} features", flush=True)

    grid, means, sds, n_resamples = _u4_tune(X_train, y_train)
    mtry = grid[_one_se_choice(means, sds, n_resamples)]
    print(f"  oneSE picked mtry={mtry}", flush=True)

    model = RandomForestClassifier(
        n_estimators=U4_N_ESTIMATORS,
        max_features=mtry,
        random_state=U4_SEED,
        n_jobs=-1,
    ).fit(X_train, y_train)
    classes = list(model.classes_)
    train_index = train.index
    proba_train = _u4_predict_proba(model, train, mask)
    del train, X_train

    test = pd.read_pickle(U4_SPLITS / "test.pkl")
    y_test = test[U4_TARGET].to_numpy()
    test_index = test.index
    proba_test = _u4_predict_proba(model, test, mask)
    del test

    return {
        "train": (pd.Series(y_train, index=train_index), proba_train, classes),
        "test": (pd.Series(y_test, index=test_index), proba_test, classes),
    }


RECIPES = {"u1": u1, "u2": u2, "u3": u3, "u4": u4}


def predictions(usecase: str) -> dict:
    """``{split: (y_true, y_pred)}`` (regression) or ``(y_true, proba, classes)``."""
    return RECIPES[usecase]()
