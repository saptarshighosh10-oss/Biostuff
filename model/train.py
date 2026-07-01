"""
Training loop stub.
Won't run until minimum data threshold is met.
"""

from model.failure_model import AggregationFailureModel, MIN_TRAINING_FAILURES


def attempt_training(labeled_failures: list, labeled_working: list):
    model = AggregationFailureModel()

    if not model.is_ready_to_train(labeled_failures):
        print(
            f"Not enough data to train yet. "
            f"Have {len(labeled_failures)} failures, need {MIN_TRAINING_FAILURES}. "
            f"Keep collecting wet-lab results."
        )
        return None

    model.train(labeled_failures, labeled_working)
    return model
