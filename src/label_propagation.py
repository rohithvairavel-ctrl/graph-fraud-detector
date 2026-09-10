"""Optional simple label propagation baseline on the account graph.

Uses sklearn.semi_supervised.LabelPropagation on a kNN graph of engineered
features (not torch-geometric). Useful as a lightweight GNN-like alternative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation

from src.config import GRAPH_FEATURES, RANDOM_STATE, TABULAR_FEATURES, TEST_SIZE


def run_label_propagation(df: pd.DataFrame) -> dict[str, float]:
    feats = TABULAR_FEATURES + [
        c for c in GRAPH_FEATURES if c != "community_id"
    ]
    X = df[feats].fillna(0.0).values
    y = df["is_fraud"].astype(int).values

    X = StandardScaler().fit_transform(X)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    y_semi = np.full(len(y), -1)
    y_semi[train_idx] = y[train_idx]

    max_n = 2500
    if len(y) > max_n:
        rng = np.random.default_rng(RANDOM_STATE)
        keep = np.concatenate(
            [
                train_idx,
                rng.choice(test_idx, size=min(len(test_idx), max_n - len(train_idx)), replace=False),
            ]
        )
        X_fit, y_fit = X[keep], y_semi[keep]
        test_mask_in_keep = np.isin(keep, test_idx)
        true_test = y[keep][test_mask_in_keep]
    else:
        X_fit, y_fit = X, y_semi
        keep = idx
        test_mask_in_keep = np.isin(keep, test_idx)
        true_test = y[test_idx]

    clf = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=1000)
    clf.fit(X_fit, y_fit)
    proba = clf.predict_proba(X_fit)[:, 1]
    scores = proba[test_mask_in_keep]

    return {
        "roc_auc": float(roc_auc_score(true_test, scores)),
        "pr_auc": float(average_precision_score(true_test, scores)),
        "model": "LabelPropagation(knn)",
        "note": "Semi-supervised propagation on standardized tabular+graph features",
    }
