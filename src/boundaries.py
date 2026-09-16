import cv2
import numpy as np


DEFAULT_ROWS = [0.55, 0.60, 0.65, 0.70, 0.75]


def _clean_mask(road_mask):
    """Clean small holes/noise in the road mask."""
    mask = (road_mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def _row_edges(mask, y):
    """Find the leftmost and rightmost road pixels on one row."""

    row = mask[y, :].astype(np.uint8)

    row_2d = row.reshape(1, -1)

    kernel = np.ones((1, 21), dtype=np.uint8)

    row_2d = cv2.morphologyEx(
        row_2d,
        cv2.MORPH_CLOSE,
        kernel
    )

    row = row_2d[0]

    xs = np.where(row > 0)[0]

    if len(xs) < 20:
        return None, None

    return int(xs[0]), int(xs[-1])


def _robust_line(points):
    """Fit x = a*y + b while removing large outliers."""

    if len(points) < 2:
        return None

    pts = np.asarray(points, dtype=np.float64)

    y = pts[:, 1]
    x = pts[:, 0]

    keep = np.ones(len(points), dtype=bool)

    for _ in range(2):
        if keep.sum() < 2:
            return None

        a, b = np.polyfit(y[keep], x[keep], 1)

        predicted = a * y + b
        residual = np.abs(x - predicted)

        median_residual = np.median(residual[keep])
        threshold = max(70.0, median_residual * 2.5)

        keep = residual <= threshold

    if keep.sum() < 2:
        return None

    a, b = np.polyfit(y[keep], x[keep], 1)

    return np.array([a, b], dtype=np.float64)


def get_road_edges_multirow(road_mask, rows=None):
    """
    Detect left/right road boundaries across multiple image rows.

    Boundary model:
        x = a*y + b
    """

    if rows is None:
        rows = DEFAULT_ROWS

    mask = _clean_mask(road_mask)

    height, width = mask.shape[:2]

    left_points = []
    right_points = []

    flags = []

    for fraction in rows:
        y = int(height * fraction)

        if y < 0 or y >= height:
            continue

        left_x, right_x = _row_edges(mask, y)

        if left_x is None or right_x is None:
            continue

        left_points.append((left_x, y))
        right_points.append((right_x, y))

    if len(left_points) < 2:
        flags.append("LEFT_EDGE_NOT_VISIBLE")

    if len(right_points) < 2:
        flags.append("RIGHT_EDGE_NOT_VISIBLE")

    left_curve = _robust_line(left_points)
    right_curve = _robust_line(right_points)

    if left_curve is None:
        flags.append("LEFT_EDGE_FIT_FAILED")

    if right_curve is None:
        flags.append("RIGHT_EDGE_FIT_FAILED")

    if not left_points or not right_points:
        flags.append("EDGE_NOT_VISIBLE")

    return {
        "left_points": left_points,
        "right_points": right_points,
        "left_curve": left_curve,
        "right_curve": right_curve,
        "flags": flags,
    }


def get_road_edges(road_mask, roi_fraction=0.70):
    """
    Backwards-compatible single-row boundary detector.
    """

    mask = _clean_mask(road_mask)

    height, width = mask.shape[:2]

    y = int(height * roi_fraction)

    left_x, right_x = _row_edges(mask, y)

    flags = []

    if left_x is None or right_x is None:
        flags.append("EDGE_NOT_VISIBLE")
        return None, width - 1, flags

    return left_x, right_x, flags
