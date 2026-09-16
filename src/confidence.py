import numpy as np


def compute_confidence(
    segmentation_score,
    left_edge_visible=True,
    right_edge_visible=True,
    temporal_stable=True,
    flags=None,
):
    """
    Compute a 0-1 confidence score for a road-width measurement.
    """

    if flags is None:
        flags = []

    score = 1.0

    # Segmentation quality
    seg = float(np.clip(segmentation_score, 0.0, 1.0))
    score *= seg

    # Boundary visibility
    if not left_edge_visible:
        score *= 0.5

    if not right_edge_visible:
        score *= 0.5

    # Temporal stability
    if not temporal_stable:
        score *= 0.6

    # Penalize important failure flags
    penalties = {
        "EDGE_NOT_VISIBLE": 0.5,
        "LEFT_EDGE_NOT_VISIBLE": 0.7,
        "RIGHT_EDGE_NOT_VISIBLE": 0.7,
        "LEFT_EDGE_FIT_FAILED": 0.6,
        "RIGHT_EDGE_FIT_FAILED": 0.6,
        "WIDTH_JUMP": 0.5,
        "INVALID_WIDTH": 0.0,
    }

    for flag in flags:
        score *= penalties.get(flag, 1.0)

    return float(np.clip(score, 0.0, 1.0))


def confidence_label(score):
    """Convert numerical confidence into a simple label."""

    score = float(score)

    if score >= 0.75:
        return "HIGH"

    if score >= 0.50:
        return "MEDIUM"

    return "LOW"
