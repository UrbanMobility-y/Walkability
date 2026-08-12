"""
Transport mode inference and walking trip refinement.

The paper uses a three class classifier (active, private, public) followed by a
high confidence active mode screen and a walking speed filter to reduce
walking/cycling/e-bike ambiguity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from spatial_utils import haversine_distance_m


FEATURE_COLUMNS = [
    "dist_OD", "dist_O_station", "dist_D_station", "dist_O_city", "dist_D_city",
    "hour", "active_t", "private_t", "public_t",
]


class TransportModeClassifier:
    """XGBoost classifier for active/private/public mode inference."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=random_state,
        )
        self.label_encoder = LabelEncoder()
        self.feature_columns: list[str] | None = None

    def train(self, X: pd.DataFrame, y: pd.Series, feature_columns: list[str] | None = None) -> dict:
        self.feature_columns = feature_columns or list(X.columns)
        y_enc = self.label_encoder.fit_transform(y)
        self.model.set_params(num_class=len(self.label_encoder.classes_))
        self.model.fit(X[self.feature_columns], y_enc)
        return {"classes": list(self.label_encoder.classes_), "n_samples": int(len(X))}

    def cross_validated_metrics(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        cols = self.feature_columns or list(X.columns)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            objective="multi:softprob", eval_metric="mlogloss", n_jobs=-1,
            random_state=self.random_state, num_class=len(le.classes_)
        )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        pred_enc = cross_val_predict(model, X[cols], y_enc, cv=skf, method="predict")
        p, r, f1, support = precision_recall_fscore_support(y_enc, pred_enc, labels=np.arange(len(le.classes_)), zero_division=0)
        return {
            "classes": list(le.classes_),
            "precision": dict(zip(le.classes_, p.round(4))),
            "recall": dict(zip(le.classes_, r.round(4))),
            "f1": dict(zip(le.classes_, f1.round(4))),
            "support": dict(zip(le.classes_, support.astype(int))),
            "confusion_matrix": confusion_matrix(y_enc, pred_enc).tolist(),
        }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns is None:
            raise RuntimeError("Classifier must be trained before prediction.")
        probs = self.model.predict_proba(X[self.feature_columns])
        pred_idx = np.argmax(probs, axis=1)
        labels = self.label_encoder.inverse_transform(pred_idx)
        out = pd.DataFrame({"pred_mode": labels}, index=X.index)
        for i, cls in enumerate(self.label_encoder.classes_):
            out[f"prob_{cls}"] = probs[:, i]
        return out


def retain_high_confidence_active(predictions: pd.DataFrame, probability_threshold: float = 0.70) -> pd.DataFrame:
    """Retain active mode trips only where P(active) exceeds the threshold."""
    if "prob_active" not in predictions.columns:
        raise ValueError("predictions must include a 'prob_active' column")
    return predictions[(predictions["pred_mode"] == "active") & (predictions["prob_active"] > probability_threshold)].copy()


def compute_mean_speed_kmh(trips: pd.DataFrame, distance_col: str = "route_length_m",
                           start_col: str = "departure_time", end_col: str = "arrival_time") -> pd.Series:
    """Compute trip mean speed in km/h from route length and observed duration."""
    start = pd.to_datetime(trips[start_col])
    end = pd.to_datetime(trips[end_col])
    duration_h = (end - start).dt.total_seconds() / 3600.0
    speed = (trips[distance_col].astype(float) / 1000.0) / duration_h.replace(0, np.nan)
    return speed.replace([np.inf, -np.inf], np.nan)


def refine_likely_walking_trips(active_trips: pd.DataFrame, speed_threshold_kmh: float = 6.0,
                                distance_col: str = "route_length_m") -> pd.DataFrame:
    """Filter active mode trips to likely walking trips using a mean speed threshold."""
    df = active_trips.copy()
    if "mean_speed_kmh" not in df.columns:
        df["mean_speed_kmh"] = compute_mean_speed_kmh(df, distance_col=distance_col)
    df["likely_walking"] = df["mean_speed_kmh"] <= speed_threshold_kmh
    return df[df["likely_walking"]].copy()


def walking_threshold_sensitivity(active_trips: pd.DataFrame, thresholds=(4, 5, 6, 7, 8, 10),
                                  distance_col: str = "route_length_m") -> pd.DataFrame:
    """Summarize walking trip retention under alternative speed thresholds."""
    df = active_trips.copy()
    if "mean_speed_kmh" not in df.columns:
        df["mean_speed_kmh"] = compute_mean_speed_kmh(df, distance_col=distance_col)
    rows = []
    for th in thresholds:
        kept = df[df["mean_speed_kmh"] <= th]
        rows.append({
            "threshold_kmh": th,
            "retained_pct": 100.0 * len(kept) / len(df) if len(df) else np.nan,
            "distance_median_m": float(kept[distance_col].median()) if len(kept) else np.nan,
            "speed_median_kmh": float(kept["mean_speed_kmh"].median()) if len(kept) else np.nan,
        })
    return pd.DataFrame(rows)


def aggregate_mode_share(predictions: pd.DataFrame) -> pd.Series:
    """Return predicted mode shares in percent."""
    return predictions["pred_mode"].value_counts(normalize=True).mul(100).sort_index()
