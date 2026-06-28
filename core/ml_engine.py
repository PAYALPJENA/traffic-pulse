"""
Module 2: Machine Learning Engine
==================================
Generates statistical predictions for traffic quantities:
  - Occupancy rate at +15 min and +30 min
  - Queue delay at +15 min and +30 min
  - Speed at +15 min and +30 min

Also provides anomaly detection (incident/blockage flagging) and
feature importance for explainability.

This module ONLY predicts — it never generates recommendations.
Recommendations are the Decision Engine's responsibility.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

# Try importing LightGBM; fall back to sklearn GradientBoosting
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    # OSError covers FileNotFoundError when the native DLL is missing on Windows
    HAS_LIGHTGBM = False

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature columns used for training
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "avg_occup_rate", "max_queue_delay", "avg_harm_speed", "total_volume",
    "avg_occup_rate_lag_1", "avg_occup_rate_lag_3", "avg_occup_rate_lag_6",
    "avg_occup_rate_roll_3_mean", "avg_occup_rate_roll_6_mean",
    "max_queue_delay_lag_1", "max_queue_delay_lag_3", "max_queue_delay_lag_6",
    "max_queue_delay_roll_3_mean", "max_queue_delay_roll_6_mean",
    "avg_harm_speed_lag_1", "avg_harm_speed_lag_3", "avg_harm_speed_lag_6",
    "avg_harm_speed_roll_3_mean", "avg_harm_speed_roll_6_mean",
    "total_volume_lag_1", "total_volume_lag_3", "total_volume_lag_6",
    "total_volume_roll_3_mean", "total_volume_roll_6_mean",
    "hour", "day_of_week", "is_weekend", "time_slot_index",
]

# What we predict
TARGET_MAP = {
    "occup_rate_15m": "avg_occup_rate_fwd_3",
    "occup_rate_30m": "avg_occup_rate_fwd_6",
    "queue_delay_15m": "max_queue_delay_fwd_3",
    "queue_delay_30m": "max_queue_delay_fwd_6",
    "speed_15m": "avg_harm_speed_fwd_3",
    "speed_30m": "avg_harm_speed_fwd_6",
}


# ---------------------------------------------------------------------------
# Congestion Predictor
# ---------------------------------------------------------------------------
class CongestionPredictor:
    """
    Ensemble regressor that predicts future traffic quantities using
    historical lag features. Uses LightGBM if available, else sklearn
    GradientBoosting.

    Attributes:
        models: dict mapping target name → fitted model
        feature_importances: dict mapping target name → {feature: importance}
        metrics: dict mapping target name → {mae, r2}
    """

    def __init__(self):
        self.models: Dict[str, object] = {}
        self.feature_importances: Dict[str, Dict[str, float]] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self._is_trained = False

    def train(self, df: pd.DataFrame, sample_frac: float = 0.15) -> None:
        """
        Train one model per prediction target.

        Args:
            df: Fully engineered DataFrame from data_processor.
            sample_frac: Fraction of data to sample for faster training.
                         Set to 1.0 for full training (slower).
        """
        # Filter to rows with all features and targets present
        all_cols = FEATURE_COLS + list(TARGET_MAP.values())
        train_df = df.dropna(subset=all_cols).copy()

        # Sample for speed if needed
        if sample_frac < 1.0 and len(train_df) > 50000:
            train_df = train_df.sample(frac=sample_frac, random_state=42)

        X = train_df[FEATURE_COLS].values
        feature_names = FEATURE_COLS

        for target_name, target_col in TARGET_MAP.items():
            y = train_df[target_col].values

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = self._create_model()

            if HAS_LIGHTGBM:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                )
            else:
                model.fit(X_train, y_train)

            # Validation metrics
            y_pred = model.predict(X_val)
            self.metrics[target_name] = {
                "mae": float(mean_absolute_error(y_val, y_pred)),
                "r2": float(r2_score(y_val, y_pred)),
            }

            # Feature importance (with fallback for models lacking the attribute)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                # Permutation-based importance fallback
                from sklearn.inspection import permutation_importance
                perm_result = permutation_importance(
                    model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1
                )
                importances = perm_result.importances_mean

            imp_dict = {name: float(max(imp, 0)) for name, imp in
                        zip(feature_names, importances)}
            # Normalize to sum=1
            total = sum(imp_dict.values()) or 1.0
            imp_dict = {k: v / total for k, v in imp_dict.items()}

            self.models[target_name] = model
            self.feature_importances[target_name] = dict(
                sorted(imp_dict.items(), key=lambda x: -x[1])
            )

        self._is_trained = True

    def predict(self, row: pd.Series) -> Dict[str, float]:
        """
        Generate predictions for a single time-point row.

        Args:
            row: A single row from the engineered DataFrame.

        Returns:
            Dict with keys: occup_rate_15m, occup_rate_30m,
            queue_delay_15m, queue_delay_30m, speed_15m, speed_30m.
        """
        if not self._is_trained:
            return self._fallback_predict(row)

        features = []
        for col in FEATURE_COLS:
            val = row.get(col, np.nan)
            features.append(float(val) if pd.notna(val) else 0.0)

        X = np.array([features])
        predictions = {}
        for target_name, model in self.models.items():
            pred = float(model.predict(X)[0])
            # Clamp to non-negative
            predictions[target_name] = max(0.0, pred)

        return predictions

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for an entire DataFrame at once (vectorized).

        Returns a DataFrame with prediction columns added.
        """
        if not self._is_trained:
            for target_name in TARGET_MAP:
                df[f"pred_{target_name}"] = np.nan
            return df

        # Build feature matrix
        X = df[FEATURE_COLS].fillna(0).values

        for target_name, model in self.models.items():
            preds = model.predict(X)
            df[f"pred_{target_name}"] = np.maximum(0.0, preds)

        return df

    def get_feature_importance(self, target: str = "occup_rate_15m") -> Dict[str, float]:
        """Return feature importance dict for the specified target."""
        return self.feature_importances.get(target, {})

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Return validation metrics for all targets."""
        return self.metrics

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    # -- Private helpers --

    def _create_model(self):
        """Create the best available gradient boosting model."""
        if HAS_LIGHTGBM:
            return lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
        else:
            return HistGradientBoostingRegressor(
                max_iter=150,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
            )

    def _fallback_predict(self, row: pd.Series) -> Dict[str, float]:
        """
        Simple fallback using rolling means when model isn't trained.
        Uses the most recent rolling average as a naive forecast.
        """
        occup = float(row.get("avg_occup_rate_roll_3_mean",
                              row.get("avg_occup_rate", 0)) or 0)
        delay = float(row.get("max_queue_delay_roll_3_mean",
                              row.get("max_queue_delay", 0)) or 0)
        speed = float(row.get("avg_harm_speed_roll_3_mean",
                              row.get("avg_harm_speed", 0)) or 0)
        return {
            "occup_rate_15m": occup,
            "occup_rate_30m": occup,
            "queue_delay_15m": delay,
            "queue_delay_30m": delay,
            "speed_15m": speed,
            "speed_30m": speed,
        }


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------
class AnomalyDetector:
    """
    Detects traffic anomalies using multi-level statistical deviation.

    Detection levels:
      1. Lane-level: speed drop AND occupancy spike simultaneously (incident)
      2. Aggregate-level: overall delay spike, speed drop, or occupancy surge
         vs. the link's historical baseline (congestion anomaly)

    This is purely statistical — it flags unusual patterns for human review.
    """

    def __init__(self, z_threshold: float = 1.5):
        self.z_threshold = z_threshold
        self._link_stats: Dict[int, Dict] = {}
        self._link_agg_stats: Dict[int, Dict] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """
        Compute per-link baseline statistics for each lane AND aggregate metrics.
        """
        for link_id, group in df.groupby("LINK_ID"):
            # Lane-level stats
            lane_stats = {}
            for lane in range(1, 7):
                speed_col = f"SPEEDAVGHARM(ALL)_{lane}"
                occup_col = f"OCCUPRATE(ALL)_{lane}"

                lane_stats[lane] = {
                    "speed_mean": group[speed_col].mean(),
                    "speed_std": group[speed_col].std(),
                    "occup_mean": group[occup_col].mean(),
                    "occup_std": group[occup_col].std(),
                }
            self._link_stats[link_id] = lane_stats

            # Aggregate-level stats (delay/speed/occupancy anomalies)
            agg = {}
            if "avg_harm_speed" in group.columns:
                agg["speed_mean"] = group["avg_harm_speed"].mean()
                agg["speed_std"] = group["avg_harm_speed"].std() or 1.0
            if "max_queue_delay" in group.columns:
                agg["delay_mean"] = group["max_queue_delay"].mean()
                agg["delay_std"] = group["max_queue_delay"].std() or 1.0
            if "avg_occup_rate" in group.columns:
                agg["occup_mean"] = group["avg_occup_rate"].mean()
                agg["occup_std"] = group["avg_occup_rate"].std() or 1.0
            self._link_agg_stats[link_id] = agg

    def detect(self, row: pd.Series) -> List[Dict]:
        """
        Check a single row for lane-level and aggregate-level anomalies.

        Returns:
            List of anomaly dicts: [{type, severity, details}, ...]
        """
        link_id = int(row.get("LINK_ID", 0))
        if link_id not in self._link_stats:
            return []

        anomalies = []

        # ── 1. Lane-level: simultaneous speed drop + occupancy spike ──
        stats = self._link_stats[link_id]
        for lane in range(1, 7):
            speed_col = f"SPEEDAVGHARM(ALL)_{lane}"
            occup_col = f"OCCUPRATE(ALL)_{lane}"

            speed = float(row.get(speed_col, 0) or 0)
            occup = float(row.get(occup_col, 0) or 0)

            lane_stats = stats[lane]
            speed_std = lane_stats["speed_std"] or 1.0
            occup_std = lane_stats["occup_std"] or 1.0

            speed_z = (lane_stats["speed_mean"] - speed) / speed_std
            occup_z = (occup - lane_stats["occup_mean"]) / occup_std

            if speed_z > self.z_threshold and occup_z > self.z_threshold:
                severity = min((speed_z + occup_z) / 2, 5.0)
                anomalies.append({
                    "lane": lane,
                    "type": "Possible Incident/Blockage",
                    "severity": round(severity, 2),
                    "details": (
                        f"Lane {lane}: speed {speed:.1f} km/h "
                        f"(avg {lane_stats['speed_mean']:.1f}), "
                        f"occupancy {occup:.2f} "
                        f"(avg {lane_stats['occup_mean']:.2f})"
                    ),
                })

        # ── 2. Aggregate-level anomalies ──
        agg = self._link_agg_stats.get(link_id, {})

        # Delay spike
        delay = float(row.get("max_queue_delay", 0) or 0)
        if "delay_mean" in agg:
            delay_z = (delay - agg["delay_mean"]) / agg["delay_std"]
            if delay_z > self.z_threshold:
                anomalies.append({
                    "type": "Delay Spike",
                    "severity": round(min(delay_z, 5.0), 2),
                    "details": (
                        f"Queue delay {delay:.0f}s is {delay_z:.1f}\u03c3 above "
                        f"baseline ({agg['delay_mean']:.0f}s avg)"
                    ),
                })

        # Speed drop
        speed_agg = float(row.get("avg_harm_speed", 0) or 0)
        if "speed_mean" in agg:
            speed_z = (agg["speed_mean"] - speed_agg) / agg["speed_std"]
            if speed_z > self.z_threshold:
                anomalies.append({
                    "type": "Unusual Speed Drop",
                    "severity": round(min(speed_z, 5.0), 2),
                    "details": (
                        f"Avg speed {speed_agg:.0f} km/h is {speed_z:.1f}\u03c3 below "
                        f"baseline ({agg['speed_mean']:.0f} km/h avg)"
                    ),
                })

        # Occupancy surge
        occup_agg = float(row.get("avg_occup_rate", 0) or 0)
        if "occup_mean" in agg:
            occup_z = (occup_agg - agg["occup_mean"]) / agg["occup_std"]
            if occup_z > self.z_threshold:
                anomalies.append({
                    "type": "Occupancy Surge",
                    "severity": round(min(occup_z, 5.0), 2),
                    "details": (
                        f"Occupancy {occup_agg:.3f} is {occup_z:.1f}\u03c3 above "
                        f"baseline ({agg['occup_mean']:.3f} avg)"
                    ),
                })

        return anomalies
