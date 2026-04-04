"""
Train ensemble model to predict WR NFL success from pre-draft features.

Ensemble of Ridge, Random Forest, and SVR with step-weighted samples
to prioritize getting top players right.

Uses normalized training data (2017-2022 WRs) and success scores.
Leave-one-year-out cross-validation to evaluate generalization.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
import logging
import os
import json
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained")

NON_FEATURE_COLS = ["name", "position", "college", "conference", "draft_year"]

# Step weights: busts 1x, mid-tier 3x, stars 8x
WEIGHT_THRESHOLDS = [(0.3, 3), (0.7, 8)]


def make_sample_weights(scores):
    """Step-weighted samples to focus on top players."""
    w = np.ones(len(scores))
    for threshold, weight in WEIGHT_THRESHOLDS:
        w[scores >= threshold] = weight
    return w


COMBINE_COLS = ["forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]


class SmartImputer(BaseEstimator, TransformerMixin):
    """Custom imputer for combine-aware NaN filling.

    - If a player has at least 1 combine measurement (attended but skipped drills),
      fill missing combine stats with 25th percentile (they probably aren't great).
    - If a player has zero combine measurements (injured/not invited),
      fill combine stats with median (don't punish for not attending).
    - Non-combine columns always filled with median.
    """

    def fit(self, X, y=None):
        self.medians_ = np.nanmedian(X, axis=0)
        self.p25_ = np.nanpercentile(X, 25, axis=0)
        # Find which column indices are combine columns
        self.combine_idx_ = []
        if hasattr(X, "columns"):
            self.combine_idx_ = [i for i, c in enumerate(X.columns) if c in COMBINE_COLS]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float).copy()
        for row_idx in range(X.shape[0]):
            has_any_combine = False
            for col_idx in self.combine_idx_:
                if not np.isnan(X[row_idx, col_idx]):
                    has_any_combine = True
                    break

            for col_idx in range(X.shape[1]):
                if np.isnan(X[row_idx, col_idx]):
                    if col_idx in self.combine_idx_ and has_any_combine:
                        X[row_idx, col_idx] = self.p25_[col_idx]
                    else:
                        X[row_idx, col_idx] = self.medians_[col_idx]
        return X


def make_pipeline(model):
    """Wrap model with smart imputer for NaN handling."""
    return Pipeline([("imp", SmartImputer()), ("m", model)])


BASE_MODELS = [
    ("ridge", lambda: Ridge(alpha=10.0)),
    ("rf", lambda: RandomForestRegressor(
        n_estimators=500, max_depth=6, min_samples_leaf=5, random_state=42)),
    ("svr", lambda: SVR(kernel="rbf", C=1.0)),
]


def load_data():
    """Load and merge normalized features with success scores."""
    features = pd.read_csv(os.path.join(PROCESSED_DIR, "normalized_wide_receiver.csv"))
    success = pd.read_csv(os.path.join(PROCESSED_DIR, "success_wide_receiver.csv"))

    df = features.merge(
        success[["name", "draft_year", "success_score"]],
        on=["name", "draft_year"],
        how="inner",
    )
    logger.info(f"Merged dataset: {len(df)} WRs, {len(df.columns)} columns")
    return df


def get_feature_cols(df):
    """Get feature column names (everything except metadata and target)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS + ["success_score"]]


def cross_validate(df, feature_cols):
    """Leave-one-year-out cross-validation with ensemble."""
    logo = LeaveOneGroupOut()
    results = []

    for train_idx, test_idx in logo.split(df, groups=df["draft_year"]):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        w = make_sample_weights(train["success_score"].values)

        preds_per_model = []
        for name, make_model in BASE_MODELS:
            m = make_pipeline(make_model())
            m.fit(train[feature_cols], train["success_score"], m__sample_weight=w)
            preds_per_model.append(m.predict(test[feature_cols]))

        ensemble_preds = np.mean(preds_per_model, axis=0)

        year = test["draft_year"].iloc[0]
        year_mae = mean_absolute_error(test["success_score"], ensemble_preds)
        logger.info(f"  {year}: MAE={year_mae:.4f} ({len(test)} WRs)")

        for n, y, a, p in zip(test["name"], test["draft_year"],
                              test["success_score"], ensemble_preds):
            results.append({"name": n, "draft_year": y, "actual": a, "predicted": p})

    rdf = pd.DataFrame(results)
    rdf["error"] = rdf["predicted"] - rdf["actual"]
    rdf["abs_error"] = rdf["error"].abs()

    mae = mean_absolute_error(rdf["actual"], rdf["predicted"])
    rmse = np.sqrt(mean_squared_error(rdf["actual"], rdf["predicted"]))
    corr = np.corrcoef(rdf["actual"], rdf["predicted"])[0, 1]
    spear, _ = spearmanr(rdf["actual"], rdf["predicted"])

    return rdf, mae, rmse, corr, spear


def train_final_models(df, feature_cols):
    """Train all base models on full data for production use."""
    w = make_sample_weights(df["success_score"].values)
    models = []
    for name, make_model in BASE_MODELS:
        m = make_pipeline(make_model())
        m.fit(df[feature_cols], df["success_score"], m__sample_weight=w)
        models.append((name, m))
        logger.info(f"  Trained {name}")
    return models


def predict(models, X):
    """Average predictions from all base models."""
    preds = [m.predict(X) for _, m in models]
    return np.mean(preds, axis=0)


if __name__ == "__main__":
    df = load_data()
    feature_cols = get_feature_cols(df)
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")

    # Cross-validate
    logger.info("Leave-one-year-out cross-validation:")
    cv_results, mae, rmse, corr, spear = cross_validate(df, feature_cols)

    print(f"\n=== Cross-Validation Results ===")
    print(f"MAE:         {mae:.4f}")
    print(f"RMSE:        {rmse:.4f}")
    print(f"Correlation: {corr:.4f}")
    print(f"Spearman:    {spear:.4f}")

    # Top 15 actual vs predicted
    cv_sorted = cv_results.sort_values("actual", ascending=False)
    print(f"\nTop 15 WRs (predicted vs actual):")
    print(f"{'Name':<25} {'Year':>5} {'Actual':>7} {'Predicted':>9} {'Error':>7}")
    print("-" * 58)
    for _, r in cv_sorted.head(15).iterrows():
        print(f"{r['name']:<25} {int(r['draft_year']):>5} {r['actual']:>7.3f} {r['predicted']:>9.3f} {r['error']:>+7.3f}")

    # Top 20 by predicted
    by_pred = cv_results.sort_values("predicted", ascending=False)
    print(f"\nTop 20 by predicted score:")
    print(f"{'#':<4} {'Name':<25} {'Year':>5} {'Actual':>7} {'Predicted':>7}")
    print("-" * 52)
    for i, (_, r) in enumerate(by_pred.head(20).iterrows()):
        print(f"{i+1:<4} {r['name']:<25} {int(r['draft_year']):>5} {r['actual']:>7.3f} {r['predicted']:>7.3f}")

    # Biggest misses
    print(f"\nBiggest misses:")
    worst = cv_results.nlargest(10, "abs_error")
    for _, r in worst.iterrows():
        print(f"  {r['name']:<25} actual={r['actual']:.3f}  predicted={r['predicted']:.3f}  error={r['error']:+.3f}")

    # Train final models on all data
    logger.info("Training final models on all data...")
    models = train_final_models(df, feature_cols)

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "ensemble_wide_receiver.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"models": models, "feature_cols": feature_cols}, f)
    logger.info(f"Models saved to {model_path}")

    # Save metadata
    meta = {
        "model_type": "ensemble",
        "base_models": [name for name, _ in BASE_MODELS],
        "sample_weighting": "step",
        "weight_thresholds": WEIGHT_THRESHOLDS,
        "features": feature_cols,
        "cv_mae": round(mae, 4),
        "cv_rmse": round(rmse, 4),
        "cv_correlation": round(corr, 4),
        "cv_spearman": round(spear, 4),
        "n_training": len(df),
    }
    meta_path = os.path.join(MODEL_DIR, "ensemble_wide_receiver_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")
