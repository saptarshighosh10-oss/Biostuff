"""
Aggregation failure predictor.
Stub — not trainable until MIN_TRAINING_FAILURES labeled examples exist.
"""

MIN_TRAINING_FAILURES = 30


class AggregationFailureModel:
    def __init__(self):
        self.trained = False
        self._weights = None

    def is_ready_to_train(self, labeled_failures: list) -> bool:
        """Guard: refuse to train until enough real labeled data exists."""
        return len(labeled_failures) >= MIN_TRAINING_FAILURES

    def train(self, labeled_failures: list, labeled_working: list):
        if not self.is_ready_to_train(labeled_failures):
            raise ValueError(
                f"Need at least {MIN_TRAINING_FAILURES} labeled failures to train. "
                f"Currently have {len(labeled_failures)}. Collect more wet-lab data first."
            )
        # training logic goes here once data exists
        raise NotImplementedError("Training not yet implemented — collect data first.")

    def predict(self, features: dict) -> float:
        """Returns aggregation risk score between 0 and 1."""
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        raise NotImplementedError
