"""
Aggregation failure predictor.
Ensemble of logistic regression + random forest trained on confirmed wet-lab failures.
Blocked until MIN_TRAINING_FAILURES labeled examples exist.
"""

import numpy as np

MIN_TRAINING_FAILURES = 30


class AggregationFailureModel:
    def __init__(self):
        self.trained = False
        self.lr = None
        self.rf = None
        self.scaler = None
        self.feature_names = None
        self.cv_scores = None
        self.n_failures = 0
        self.n_working = 0

    def is_ready_to_train(self, labeled_failures: list) -> bool:
        return len(labeled_failures) >= MIN_TRAINING_FAILURES

    def train(self, X: list[list[float]], y: list[int], feature_names: list[str]):
        """
        Train on feature matrix X (n_samples × n_features) and binary labels y.
        y=1 means confirmed_failure, y=0 means working.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold, cross_val_score

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

        # scale for logistic regression
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        self.rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

        cv = StratifiedKFold(n_splits=min(5, self.n_failures), shuffle=True, random_state=42)
        lr_scores = cross_val_score(self.lr, X_scaled, y, cv=cv, scoring="roc_auc")
        rf_scores = cross_val_score(self.rf, X, y, cv=cv, scoring="roc_auc")

        self.cv_scores = {
            "logistic_regression_auc": lr_scores,
            "random_forest_auc": rf_scores,
        }

        self.lr.fit(X_scaled, y)
        self.rf.fit(X, y)
        self.trained = True

    def predict_proba(self, X: list[list[float]]) -> np.ndarray:
        """Returns failure probability for each sample (ensemble average)."""
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        X = np.array(X, dtype=float)
        X_scaled = self.scaler.transform(X)
        lr_proba = self.lr.predict_proba(X_scaled)[:, 1]
        rf_proba = self.rf.predict_proba(X)[:, 1]
        return (lr_proba + rf_proba) / 2.0

    def predict_with_uncertainty(self, X: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (ensemble_prob, confidence_gap) for each sample.
        confidence_gap = abs(LR_prob - RF_prob): 0 = both models agree, 1 = maximum disagreement.
        High gap + high risk = uncertain but concerning = highest wet-lab value.
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        X = np.array(X, dtype=float)
        X_scaled = self.scaler.transform(X)
        lr_proba = self.lr.predict_proba(X_scaled)[:, 1]
        rf_proba = self.rf.predict_proba(X)[:, 1]
        ensemble = (lr_proba + rf_proba) / 2.0
        gap = np.abs(lr_proba - rf_proba)
        return ensemble, gap

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
