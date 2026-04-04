"""
Experiment with different model configurations for WR and RB.
Tests various base models, hyperparameters, and ensemble combos.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
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


class SmartImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.medians_ = np.nanmedian(X, axis=0)
        self.p25_ = np.nanpercentile(X, 25, axis=0)
        self.combine_idx_ = []
        self.stat_idx_ = []
        if hasattr(X, "columns"):
            self.combine_idx_ = [i for i, c in enumerate(X.columns) if c in COMBINE_COLS]
            self.stat_idx_ = [i for i, c in enumerate(X.columns)
                              if c[:3] in ("Y0_", "Y1_", "Y2_", "Y3_")]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float).copy()
        for row_idx in range(X.shape[0]):
            has_any_combine = any(not np.isnan(X[row_idx, ci]) for ci in self.combine_idx_)
            for col_idx in range(X.shape[1]):
                if np.isnan(X[row_idx, col_idx]):
                    if col_idx in self.stat_idx_:
                        X[row_idx, col_idx] = 0.0
                    elif col_idx in self.combine_idx_ and has_any_combine:
                        X[row_idx, col_idx] = self.p25_[col_idx]
                    else:
                        X[row_idx, col_idx] = self.medians_[col_idx]
        return X


def load_data(position_label):
    features = pd.read_csv(os.path.join(PROCESSED_DIR, f"normalized_{position_label}.csv"))
    success = pd.read_csv(os.path.join(PROCESSED_DIR, f"success_{position_label}.csv"))
    df = features.merge(success[["name", "draft_year", "success_score"]], on=["name", "draft_year"], how="inner")
    return df


def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS + ["success_score"]]


def cv_eval(df, feature_cols, models_config):
    """Run LOGO CV with given model config, return metrics."""
    logo = LeaveOneGroupOut()
    all_actual, all_pred = [], []

    for train_idx, test_idx in logo.split(df, groups=df["draft_year"]):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        w = make_sample_weights(train["success_score"].values)

        preds_per_model = []
        for make_model in models_config:
            m = Pipeline([("imp", SmartImputer()), ("m", make_model())])
            m.fit(train[feature_cols], train["success_score"], m__sample_weight=w)
            preds_per_model.append(m.predict(test[feature_cols]))

        ensemble_preds = np.mean(preds_per_model, axis=0)
        all_actual.extend(test["success_score"].values)
        all_pred.extend(ensemble_preds)

    actual, pred = np.array(all_actual), np.array(all_pred)
    mae = mean_absolute_error(actual, pred)
    corr = np.corrcoef(actual, pred)[0, 1]
    spear, _ = spearmanr(actual, pred)
    return mae, corr, spear


# Model configs to test
EXPERIMENTS = {
    "Ridge(10)": [lambda: Ridge(alpha=10.0)],
    "Ridge(1)": [lambda: Ridge(alpha=1.0)],
    "Ridge(50)": [lambda: Ridge(alpha=50.0)],
    "RF(500,d6)": [lambda: RandomForestRegressor(500, max_depth=6, min_samples_leaf=5, random_state=42)],
    "RF(500,d4)": [lambda: RandomForestRegressor(500, max_depth=4, min_samples_leaf=5, random_state=42)],
    "RF(500,d8)": [lambda: RandomForestRegressor(500, max_depth=8, min_samples_leaf=3, random_state=42)],
    "SVR(rbf,C=1)": [lambda: SVR(kernel="rbf", C=1.0)],
    "SVR(rbf,C=10)": [lambda: SVR(kernel="rbf", C=10.0)],
    "SVR(rbf,C=0.1)": [lambda: SVR(kernel="rbf", C=0.1)],
    "GBR(100,d3)": [lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)],
    "GBR(200,d2)": [lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42)],
    "ExtraTrees(500,d6)": [lambda: ExtraTreesRegressor(500, max_depth=6, min_samples_leaf=5, random_state=42)],
    "Lasso(0.01)": [lambda: Lasso(alpha=0.01)],
    "ElasticNet(0.01)": [lambda: ElasticNet(alpha=0.01, l1_ratio=0.5)],

    # Ensembles
    "Ridge+RF+SVR": [
        lambda: Ridge(alpha=10.0),
        lambda: RandomForestRegressor(500, max_depth=6, min_samples_leaf=5, random_state=42),
        lambda: SVR(kernel="rbf", C=1.0),
    ],
    "Ridge+RF+GBR": [
        lambda: Ridge(alpha=10.0),
        lambda: RandomForestRegressor(500, max_depth=6, min_samples_leaf=5, random_state=42),
        lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    ],
    "Ridge+GBR+SVR": [
        lambda: Ridge(alpha=10.0),
        lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        lambda: SVR(kernel="rbf", C=1.0),
    ],
    "RF+GBR+ExtraTrees": [
        lambda: RandomForestRegressor(500, max_depth=6, min_samples_leaf=5, random_state=42),
        lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        lambda: ExtraTreesRegressor(500, max_depth=6, min_samples_leaf=5, random_state=42),
    ],
    "All5": [
        lambda: Ridge(alpha=10.0),
        lambda: RandomForestRegressor(500, max_depth=6, min_samples_leaf=5, random_state=42),
        lambda: SVR(kernel="rbf", C=1.0),
        lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        lambda: ExtraTreesRegressor(500, max_depth=6, min_samples_leaf=5, random_state=42),
    ],
}


if __name__ == "__main__":
    for pos in ["wide_receiver", "running_back"]:
        df = load_data(pos)
        feature_cols = get_feature_cols(df)
        print(f"\n{'='*60}")
        print(f"  {pos.upper()} ({len(df)} players, {len(feature_cols)} features)")
        print(f"{'='*60}")
        print(f"{'Config':<25} {'MAE':>7} {'Corr':>7} {'Spear':>7}")
        print("-" * 50)

        results = []
        for name, config in EXPERIMENTS.items():
            mae, corr, spear = cv_eval(df, feature_cols, config)
            results.append((name, mae, corr, spear))
            print(f"{name:<25} {mae:>7.4f} {corr:>7.4f} {spear:>7.4f}")

        # Sort by correlation
        results.sort(key=lambda x: -x[2])
        print(f"\nRanked by correlation:")
        for i, (name, mae, corr, spear) in enumerate(results):
            print(f"  {i+1}. {name:<25} corr={corr:.4f}  spear={spear:.4f}  mae={mae:.4f}")
