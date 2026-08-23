"""mAML's published scaler and classifier search space.

Transcribed from `yangfenglong/mAML1.0@master:code/sklearn_pipeline_config.py`
lines 31-109 (read 2026-08-23), with `NonScaler` from the same repository's
`code/utils.py` lines 116-123.

Two values are substituted because they no longer construct under
scikit-learn 1.5:

- `LogisticRegression(..., multi_class='auto')` -- the argument was removed;
  dropped here, its behaviour is now the only behaviour.
- `SGDClassifier` grid value `loss='log'` -- renamed upstream to `'log_loss'`.

Everything else, including the parameter grids and `RANDOM_STATE = 0`, is
reproduced verbatim.
"""

from __future__ import annotations

import numpy as np
from lightgbm.sklearn import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

RANDOM_STATE = 0


class NonScaler(BaseEstimator, TransformerMixin):
    """sklearn pipeline did nothing transform"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


SCALERS = [
    ("Non", NonScaler()),
    ("Binarizer", Binarizer(threshold=0)),
    ("MinMaxScaler", MinMaxScaler()),
    ("MaxAbsScaler", MaxAbsScaler()),
    ("StandardScaler", StandardScaler()),
    ("RobustScaler", RobustScaler(quantile_range=(25, 75), with_centering=False)),
    (
        "PowerTransformer_YeoJohnson",
        PowerTransformer(method="yeo-johnson", standardize=False),
    ),
    ("QuantileTransformer_Normal", QuantileTransformer(output_distribution="normal")),
    ("QuantileTransformer_Uniform", QuantileTransformer(output_distribution="uniform")),
    ("Normalizer", Normalizer()),
    ("Log1p", FunctionTransformer(np.log1p, validate=False)),
]

TREE_BASED_CLASSIFIERS = [
    (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        dict(clf__max_depth=list(map(int, np.logspace(2, 6, 5, base=2)))),
    ),
    (
        BaggingClassifier(random_state=RANDOM_STATE),
        dict(clf__n_estimators=list(map(int, np.linspace(5, 50, 10)))),
    ),
    (
        GradientBoostingClassifier(),
        dict(clf__learning_rate=[0.001, 0.01, 0.1, 0.2, 0.5]),
    ),
    (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        dict(clf__learning_rate=[0.001, 0.01, 0.1, 0.2, 0.5]),
    ),
    (
        RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE),
        dict(clf__max_depth=list(map(int, np.logspace(2, 6, 5, base=2)))),
    ),
    (
        ExtraTreesClassifier(random_state=RANDOM_STATE),
        dict(clf__max_depth=list(map(int, np.logspace(2, 6, 5, base=2)))),
    ),
    (
        XGBClassifier(random_state=RANDOM_STATE),
        dict(
            clf__max_depth=list(map(int, np.logspace(2, 6, 5, base=2))),
            clf__min_child_weight=range(1, 6, 2),
        ),
    ),
    (
        LGBMClassifier(random_state=RANDOM_STATE),
        dict(clf__max_depth=list(map(int, np.logspace(5, 6, 5, base=2)))),
    ),
]

OTHER_CLASSIFIERS = [
    (
        KNeighborsClassifier(),
        dict(clf__n_neighbors=list(map(int, np.linspace(5, 20, 4)))),
    ),
    (GaussianNB(), dict()),
    (
        LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.15,
            solver="saga",
            random_state=RANDOM_STATE,
        ),
        dict(clf__C=list(np.logspace(-4, 4, 3))),
    ),
    (
        LinearSVC(random_state=RANDOM_STATE),
        dict(clf__C=list(np.logspace(-4, 4, 3))),
    ),
    # Upstream comments out MLPClassifier:
    #    (MLPClassifier(max_iter=10000, random_state=RANDOM_STATE),
    #     dict(clf__alpha=[0.1, 0.001, 0.0001],
    #          clf__solver=["lbfgs", "sgd", "adam"])),
    (
        SGDClassifier(penalty="elasticnet", l1_ratio=0.15, random_state=RANDOM_STATE),
        dict(
            clf__loss=[
                "hinge",
                "log_loss",
                "modified_huber",
                "squared_hinge",
                "perceptron",
            ]
        ),
    ),
]

ALL_CLASSIFIERS = TREE_BASED_CLASSIFIERS + OTHER_CLASSIFIERS
