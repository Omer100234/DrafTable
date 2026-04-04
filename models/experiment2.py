"""More tree/forest experiments for WR and RB."""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
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
            self.stat_idx_ = [i for i, c in enumerate(X.columns) if c[:3] in ("Y0_", "Y1_", "Y2_", "Y3_")]
        return self
    def transform(self, X):
        X = np.array(X, dtype=float).copy()
        for row_idx in range(X.shape[0]):
            has_any_combine = any(not np.isnan(X[row_idx, ci]) for ci in self.combine_idx_)
            for col_idx in range(X.shape[1]):
                if np.isnan(X[row_idx, col_idx]):
                    if col_idx in self.stat_idx_:
                        X[row_idx, col_idx] = self.p25_[col_idx]
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

def cv_eval(df, feature_cols, models_config):
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
        all_actual.extend(test["success_score"].values)
        all_pred.extend(np.mean(preds_per_model, axis=0))
    actual, pred = np.array(all_actual), np.array(all_pred)
    return mean_absolute_error(actual, pred), np.corrcoef(actual, pred)[0,1], spearmanr(actual, pred)[0]

EXPERIMENTS = {
    # Single trees/forests
    "RF(300,d5,leaf3)": [lambda: RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=3, random_state=42)],
    "RF(500,d8,leaf3)": [lambda: RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42)],
    "RF(500,d10,leaf3)": [lambda: RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=3, random_state=42)],
    "RF(1000,d6,leaf5)": [lambda: RandomForestRegressor(n_estimators=1000, max_depth=6, min_samples_leaf=5, random_state=42)],
    "RF(1000,d8,leaf3)": [lambda: RandomForestRegressor(n_estimators=1000, max_depth=8, min_samples_leaf=3, random_state=42)],
    "ET(500,d8,leaf3)": [lambda: ExtraTreesRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42)],
    "ET(1000,d6,leaf5)": [lambda: ExtraTreesRegressor(n_estimators=1000, max_depth=6, min_samples_leaf=5, random_state=42)],
    "GBR(200,d3,lr.05)": [lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)],
    "GBR(300,d2,lr.03)": [lambda: GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, random_state=42)],
    "GBR(500,d2,lr.01)": [lambda: GradientBoostingRegressor(n_estimators=500, max_depth=2, learning_rate=0.01, random_state=42)],
    "GBR(100,d4,lr.1)": [lambda: GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)],

    # Ensembles with Ridge(50) since it's best linear for WR
    "Ridge50+RF(500,d8)": [
        lambda: Ridge(alpha=50.0),
        lambda: RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42),
    ],
    "Ridge50+RF(500,d6)+GBR(200,d3)": [
        lambda: Ridge(alpha=50.0),
        lambda: RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_leaf=5, random_state=42),
        lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    ],
    "Ridge50+RF(500,d8)+GBR(200,d3)": [
        lambda: Ridge(alpha=50.0),
        lambda: RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42),
        lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    ],
    "Ridge10+RF(500,d8)+GBR(200,d3)": [
        lambda: Ridge(alpha=10.0),
        lambda: RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42),
        lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    ],
    "Ridge50+RF(1000,d8)+GBR(300,d2)": [
        lambda: Ridge(alpha=50.0),
        lambda: RandomForestRegressor(n_estimators=1000, max_depth=8, min_samples_leaf=3, random_state=42),
        lambda: GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, random_state=42),
    ],
    "Ridge50+ET(500,d8)+GBR(200,d3)": [
        lambda: Ridge(alpha=50.0),
        lambda: ExtraTreesRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42),
        lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    ],
    "Ridge50+RF(500,d8)+ET(500,d8)": [
        lambda: Ridge(alpha=50.0),
        lambda: RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42),
        lambda: ExtraTreesRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42),
    ],
    "Ridge50+RF+GBR+ET": [
        lambda: Ridge(alpha=50.0),
        lambda: RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42),
        lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
        lambda: ExtraTreesRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42),
    ],
    # Previous best refs
    "Ridge(50) [ref]": [lambda: Ridge(alpha=50.0)],
    "Ridge+RF+GBR [ref]": [
        lambda: Ridge(alpha=10.0),
        lambda: RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_leaf=5, random_state=42),
        lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    ],
}

if __name__ == "__main__":
    for pos in ["wide_receiver", "running_back"]:
        df = load_data(pos)
        fc = get_feature_cols(df)
        print(f"\n{'='*65}")
        print(f"  {pos.upper()} ({len(df)} players, {len(fc)} features)")
        print(f"{'='*65}")
        print(f"{'Config':<35} {'MAE':>7} {'Corr':>7} {'Spear':>7}")
        print("-" * 60)
        results = []
        for name, config in EXPERIMENTS.items():
            mae, corr, spear = cv_eval(df, fc, config)
            results.append((name, mae, corr, spear))
            print(f"{name:<35} {mae:>7.4f} {corr:>7.4f} {spear:>7.4f}")
        results.sort(key=lambda x: -x[2])
        print(f"\nRanked by correlation:")
        for i, (name, mae, corr, spear) in enumerate(results):
            print(f"  {i+1:>2}. {name:<35} corr={corr:.4f}  spear={spear:.4f}  mae={mae:.4f}")
