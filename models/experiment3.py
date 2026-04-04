"""Test different stat fill percentiles for WR vs RB."""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
NON_FEATURE_COLS = ["name", "position", "college", "conference", "draft_year"]
COMBINE_COLS = ["forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
WEIGHT_THRESHOLDS = [(0.3, 3), (0.7, 8)]

def make_sample_weights(scores):
    w = np.ones(len(scores))
    for threshold, weight in WEIGHT_THRESHOLDS:
        w[scores >= threshold] = weight
    return w

class PercentileImputer(BaseEstimator, TransformerMixin):
    def __init__(self, stat_percentile=25):
        self.stat_percentile = stat_percentile

    def fit(self, X, y=None):
        self.medians_ = np.nanmedian(X, axis=0)
        self.p25_ = np.nanpercentile(X, 25, axis=0)
        self.stat_fill_ = np.nanpercentile(X, self.stat_percentile, axis=0)
        self.combine_idx_ = []
        self.stat_idx_ = []
        if hasattr(X, "columns"):
            self.combine_idx_ = [i for i, c in enumerate(X.columns) if c in COMBINE_COLS]
            self.stat_idx_ = [i for i, c in enumerate(X.columns) if c[:3] in ("Y0_", "Y1_", "Y2_", "Y3_")]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float).copy()
        for row_idx in range(X.shape[0]):
            has_any_combine = any(not np.isnan(X[row_idx, ci]) for ci in self.combine_idx_)
            for col_idx in range(X.shape[1]):
                if np.isnan(X[row_idx, col_idx]):
                    if col_idx in self.stat_idx_:
                        X[row_idx, col_idx] = self.stat_fill_[col_idx]
                    elif col_idx in self.combine_idx_ and has_any_combine:
                        X[row_idx, col_idx] = self.p25_[col_idx]
                    else:
                        X[row_idx, col_idx] = self.medians_[col_idx]
        return X

def load_data(pos):
    f = pd.read_csv(os.path.join(PROCESSED_DIR, f"normalized_{pos}.csv"))
    s = pd.read_csv(os.path.join(PROCESSED_DIR, f"success_{pos}.csv"))
    return f.merge(s[["name","draft_year","success_score"]], on=["name","draft_year"], how="inner")

def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS + ["success_score"]]

def cv_eval(df, feature_cols, models_config, stat_pct):
    logo = LeaveOneGroupOut()
    all_actual, all_pred = [], []
    for train_idx, test_idx in logo.split(df, groups=df["draft_year"]):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        w = make_sample_weights(train["success_score"].values)
        preds_per_model = []
        for make_model in models_config:
            m = Pipeline([("imp", PercentileImputer(stat_percentile=stat_pct)), ("m", make_model())])
            m.fit(train[feature_cols], train["success_score"], m__sample_weight=w)
            preds_per_model.append(m.predict(test[feature_cols]))
        all_actual.extend(test["success_score"].values)
        all_pred.extend(np.mean(preds_per_model, axis=0))
    actual, pred = np.array(all_actual), np.array(all_pred)
    return mean_absolute_error(actual, pred), np.corrcoef(actual, pred)[0,1], spearmanr(actual, pred)[0]

# Best models from previous experiments
WR_MODEL = [
    lambda: Ridge(alpha=10.0),
    lambda: RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_leaf=5, random_state=42),
    lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
]

RB_MODEL = [
    lambda: Ridge(alpha=50.0),
    lambda: RandomForestRegressor(n_estimators=1000, max_depth=8, min_samples_leaf=3, random_state=42),
    lambda: GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, random_state=42),
]

PERCENTILES = [0, 10, 15, 20, 25, 30, 35, 40, 50]

if __name__ == "__main__":
    for pos, model in [("wide_receiver", WR_MODEL), ("running_back", RB_MODEL)]:
        df = load_data(pos)
        fc = get_feature_cols(df)
        print(f"\n{'='*55}")
        print(f"  {pos.upper()} ({len(df)} players)")
        print(f"{'='*55}")
        print(f"{'Percentile':>12} {'MAE':>7} {'Corr':>7} {'Spear':>7}")
        print("-" * 40)
        for pct in PERCENTILES:
            mae, corr, spear = cv_eval(df, fc, model, pct)
            print(f"{pct:>10}th {mae:>7.4f} {corr:>7.4f} {spear:>7.4f}")
