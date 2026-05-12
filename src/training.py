from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import FEATURE_COLS, TARGET_COL


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def split_train_test(df: pd.DataFrame):
    x = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int).copy()
    return train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)


def find_best_threshold_for_recall(y_true: pd.Series, proba_risk: np.ndarray):
    best_threshold = 0.5
    best_recall = -1.0
    best_precision = 0.0
    best_f1 = 0.0

    for thr in np.arange(0.30, 0.71, 0.01):
        pred = (proba_risk >= thr).astype(int)
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if recall > best_recall or (recall == best_recall and f1 > best_f1):
            best_threshold = float(np.round(thr, 2))
            best_recall = recall
            best_precision = precision
            best_f1 = f1

    return {
        "threshold": best_threshold,
        "recall": best_recall,
        "precision": best_precision,
        "f1": best_f1,
    }


def cross_validated_predictions(model: Pipeline, x: pd.DataFrame, y: pd.Series, folds: int = 5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    pred_proba = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    return pred_proba


def save_model_bundle(model: Pipeline, threshold: float, output_path: str):
    bundle = {
        "model": model,
        "threshold": threshold,
        "features": FEATURE_COLS,
        "target": TARGET_COL,
    }
    joblib.dump(bundle, output_path)


def load_model_bundle(path: str):
    return joblib.load(path)
