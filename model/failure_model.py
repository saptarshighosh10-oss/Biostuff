"""
Aggregation failure predictor.
Ensemble of logistic regression + random forest trained on confirmed wet-lab failures.
Blocked until MIN_TRAINING_FAILURES labeled examples exist.
"""

import numpy as np

MIN_TRAINING_FAILURES = 30
MAX_REFERENCE_SAMPLES = 5000  # cap on stored nearest-neighbor reference set


class AggregationFailureModel:
    def __init__(self):
        self.trained = False
        self.lr = None
        self.rf = None
        self.lr_calibrated = None
        self.rf_calibrated = None
        self.calibration_method = None
        self.scaler = None
        self.feature_names = None
        self.cv_scores = None
        self.used_grouped_cv = False
        self.n_groups = None
        self.n_failures = 0
        self.n_working = 0
        self.reference_X = None
        self.reference_y = None
        self.reference_identities = None

    def is_ready_to_train(self, labeled_failures: list) -> bool:
        return len(labeled_failures) >= MIN_TRAINING_FAILURES

    def train(self, X: list[list[float]], y: list[int], feature_names: list[str],
              groups: list | None = None, identities: list[dict] | None = None):
        """
        Train on feature matrix X (n_samples × n_features) and binary labels y.
        y=1 means confirmed_failure, y=0 means working.

        groups: optional per-sample group key (e.g. wild-type/assay ID or source
        dataset). When provided with >=2 distinct groups, cross-validation uses
        StratifiedGroupKFold so mutants of the same protein never split across
        train/test folds — without this, near-duplicate variants leak between
        folds and CV AUC is optimistically biased. Falls back to plain
        StratifiedKFold when groups are absent or too few to split on.

        identities: optional per-sample metadata dicts ({"name","source","sequence"})
        used to power nearest_neighbors() lookups at inference time — "this
        candidate most resembles known-working protein X". Stored capped at
        MAX_REFERENCE_SAMPLES (random subsample) to keep the saved model small.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        if y.sum() < MIN_TRAINING_FAILURES:
            raise ValueError(
                f"Need at least {MIN_TRAINING_FAILURES} failure examples. "
                f"Currently have {int(y.sum())}. Collect more wet-lab data first."
            )

        self.feature_names = feature_names
        self.n_failures = int(y.sum())
        self.n_working = int((y == 0).sum())

        # ── nearest-neighbor reference index ──────────────────────────────
        # Store a (capped, randomly subsampled) copy of the raw training
        # features + identities so predict-time code can report "this
        # candidate most resembles known-working/failure protein X".
        if identities is not None:
            n = len(X)
            if n > MAX_REFERENCE_SAMPLES:
                rng = np.random.RandomState(42)
                idx = rng.choice(n, size=MAX_REFERENCE_SAMPLES, replace=False)
            else:
                idx = np.arange(n)
            self.reference_X = X[idx].copy()
            self.reference_y = y[idx].copy()
            self.reference_identities = [identities[i] for i in idx]

        # scale for logistic regression
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        self.rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

        # ── choose grouped vs standard CV splitter ────────────────────────
        groups_arr = np.array(groups) if groups is not None else None
        n_unique_groups = len(np.unique(groups_arr)) if groups_arr is not None else 0
        n_splits = min(5, self.n_failures)

        if groups_arr is not None and n_unique_groups >= 2 and n_unique_groups >= n_splits:
            cv = StratifiedGroupKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)
            cv_splits = list(cv.split(X_scaled, y, groups=groups_arr))
            self.used_grouped_cv = True
            self.n_groups = n_unique_groups
        else:
            cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)
            cv_splits = list(cv.split(X_scaled, y))
            self.used_grouped_cv = False
            self.n_groups = n_unique_groups if groups_arr is not None else None

        lr_scores = cross_val_score(self.lr, X_scaled, y, cv=cv_splits, scoring="roc_auc")
        rf_scores = cross_val_score(self.rf, X, y, cv=cv_splits, scoring="roc_auc")

        self.cv_scores = {
            "logistic_regression_auc": lr_scores,
            "random_forest_auc": rf_scores,
        }

        self.lr.fit(X_scaled, y)
        self.rf.fit(X, y)

        # ── probability calibration ────────────────────────────────────────
        # Raw LR/RF probabilities are not reliably calibrated (RF especially
        # tends toward overconfidence near 0/1). Wrap each in a CV-calibrated
        # classifier using the SAME fold splits as above, so "0.7" means the
        # model is actually right about 70% of the time on held-out folds.
        # Isotonic needs more data than sigmoid/Platt to avoid overfitting.
        self.calibration_method = "isotonic" if len(y) >= 1000 else "sigmoid"
        self.lr_calibrated = CalibratedClassifierCV(self.lr, method=self.calibration_method, cv=cv_splits)
        self.lr_calibrated.fit(X_scaled, y)
        self.rf_calibrated = CalibratedClassifierCV(self.rf, method=self.calibration_method, cv=cv_splits)
        self.rf_calibrated.fit(X, y)

        self.trained = True

    def predict_proba(self, X: list[list[float]]) -> np.ndarray:
        """Returns calibrated failure probability for each sample (ensemble average)."""
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        X = np.array(X, dtype=float)
        X_scaled = self.scaler.transform(X)
        # getattr guards against model.pkl files saved before calibration was
        # added — their __dict__ has no lr_calibrated/rf_calibrated key.
        lr_model = getattr(self, "lr_calibrated", None) or self.lr
        rf_model = getattr(self, "rf_calibrated", None) or self.rf
        lr_proba = lr_model.predict_proba(X_scaled)[:, 1]
        rf_proba = rf_model.predict_proba(X)[:, 1]
        return (lr_proba + rf_proba) / 2.0

    def predict_with_uncertainty(self, X: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (ensemble_prob, confidence_gap) for each sample, using calibrated
        probabilities. confidence_gap = abs(LR_prob - RF_prob): 0 = both models
        agree, 1 = maximum disagreement. High gap + high risk = uncertain but
        concerning = highest wet-lab value.
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        X = np.array(X, dtype=float)
        X_scaled = self.scaler.transform(X)
        lr_model = getattr(self, "lr_calibrated", None) or self.lr
        rf_model = getattr(self, "rf_calibrated", None) or self.rf
        lr_proba = lr_model.predict_proba(X_scaled)[:, 1]
        rf_proba = rf_model.predict_proba(X)[:, 1]
        ensemble = (lr_proba + rf_proba) / 2.0
        gap = np.abs(lr_proba - rf_proba)
        return ensemble, gap

    def nearest_neighbors(self, x: list[float], label: int | None = None, top_k: int = 3) -> list[dict]:
        """
        Find the closest stored training examples to feature vector x, in
        scaled feature space. label=0 restricts the search to 'working'
        references, label=1 to 'confirmed_failure', label=None searches both.
        Returns a list of {rank, name, source, label, distance}, nearest first.
        Returns [] if the model has no stored reference index (e.g. an old
        model.pkl saved before this feature was added).
        """
        if not self.trained or self.reference_X is None or not len(self.reference_X):
            return []

        X_ref = np.asarray(self.reference_X, dtype=float)
        y_ref = np.asarray(self.reference_y, dtype=int)
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)

        mask = np.ones(len(y_ref), dtype=bool) if label is None else (y_ref == label)
        if not mask.any():
            return []

        X_ref_scaled = self.scaler.transform(X_ref[mask])
        x_scaled = self.scaler.transform(x_arr)
        dists = np.linalg.norm(X_ref_scaled - x_scaled, axis=1)

        subset_identities = [ident for ident, keep in zip(self.reference_identities, mask) if keep]
        subset_labels = y_ref[mask]

        order = np.argsort(dists)[:top_k]
        results = []
        for rank, idx in enumerate(order):
            ident = subset_identities[idx]
            results.append({
                "rank": rank + 1,
                "name": ident.get("name", ""),
                "source": ident.get("source", ""),
                "label": "confirmed_failure" if subset_labels[idx] == 1 else "working",
                "distance": round(float(dists[idx]), 4),
            })
        return results

    def predict(self, features: dict) -> float:
        """Score a single candidate dict. Returns failure probability 0-1."""
        from model.features import features_to_vector, FEATURE_NAMES
        vec = features_to_vector(features)
        return float(self.predict_proba([vec])[0])

    def feature_importance(self) -> list[tuple[str, float]]:
        """
        Returns (feature_name, importance) sorted descending.
        Combines |LR coefficients| + RF importances (normalized separately, then averaged).
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet.")

        lr_coefs = np.abs(self.lr.coef_[0])
        lr_norm = lr_coefs / (lr_coefs.sum() + 1e-9)

        rf_imp = self.rf.feature_importances_
        rf_norm = rf_imp / (rf_imp.sum() + 1e-9)

        combined = (lr_norm + rf_norm) / 2.0
        pairs = sorted(zip(self.feature_names, combined), key=lambda x: x[1], reverse=True)
        return [(name, round(float(score), 4)) for name, score in pairs]

    def save(self, path: str):
        import joblib
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AggregationFailureModel":
        import joblib
        return joblib.load(path)
