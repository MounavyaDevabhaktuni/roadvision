import numpy as np
import cv2


EDGE_NOT_VISIBLE = "EDGE_NOT_VISIBLE"
EDGE_JUMP = "EDGE_JUMP"
OCCLUSION = "OCCLUSION"

DEFAULT_ROWS = [
    0.55,
    0.58,
    0.61,
    0.64,
    0.67,
    0.70,
    0.73,
    0.75,
]


def _clean_mask(road_mask):
    mask = (road_mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


def _row_edges(mask, y):
    h, w = mask.shape

    if y < 0 or y >= h:
        return None

    row = mask[y].copy()

    # Close small gaps in the road mask.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (31, 1)
    )

    row = cv2.morphologyEx(
        row.reshape(1, -1),
        cv2.MORPH_CLOSE,
        kernel
    )[0]

    xs = np.flatnonzero(row > 0)

    if len(xs) < 20:
        return None

    # Find continuous regions.
    runs = []

    start = xs[0]
    previous = xs[0]

    for x in xs[1:]:
        if x > previous + 1:
            if previous - start + 1 >= 20:
                runs.append((start, previous))
            start = x

        previous = x

    if previous - start + 1 >= 20:
        runs.append((start, previous))

    if not runs:
        return None

    # Pick the widest substantial road region.
    left, right = max(
        runs,
        key=lambda r: r[1] - r[0]
    )

    if right - left < w * 0.20:
        return None

    return int(left), int(right)


def _remove_jumps(points, max_jump=90):
    """
    Remove isolated boundary jumps.

    The road boundary should move reasonably smoothly
    from one image row to the next.
    """

    if len(points) < 3:
        return points

    points = sorted(
        points,
        key=lambda p: p[1]
    )

    result = [points[0]]

    for point in points[1:]:

        previous_x = result[-1][0]

        if abs(point[0] - previous_x) <= max_jump:
            result.append(point)

    return result


def _fit_boundary(points):
    """
    Fit x = a*y + b using RANSAC-like pair testing.

    This prevents one bad segmentation row from
    bending the boundary.
    """

    if len(points) < 3:
        return None

    points = np.asarray(
        points,
        dtype=float
    )

    best_inliers = None
    best_count = -1
    best_error = float("inf")

    n = len(points)

    # Try every pair as a candidate line.
    for i in range(n):
        for j in range(i + 1, n):

            y1 = points[i, 1]
            x1 = points[i, 0]

            y2 = points[j, 1]
            x2 = points[j, 0]

            if y2 == y1:
                continue

            a = (x2 - x1) / (y2 - y1)
            b = x1 - a * y1

            predicted = a * points[:, 1] + b

            residual = np.abs(
                points[:, 0] - predicted
            )

            inliers = residual <= 35

            count = int(inliers.sum())
            error = float(residual[inliers].sum())

            if (
                count > best_count
                or (
                    count == best_count
                    and error < best_error
                )
            ):
                best_count = count
                best_error = error
                best_inliers = inliers

    if best_inliers is None or best_count < 3:
        return None

    # Refit using only the good points.
    good = points[best_inliers]

    a, b = np.polyfit(
        good[:, 1],
        good[:, 0],
        1
    )

    return np.array([a, b])


def get_road_edges(
    road_mask,
    roi_fraction=0.70
):

    mask = _clean_mask(road_mask)

    h, _ = mask.shape

    y = int(
        roi_fraction * h
    )

    result = _row_edges(
        mask,
        y
    )

    if result is None:
        return (
            None,
            None,
            [EDGE_NOT_VISIBLE]
        )

    return (
        result[0],
        result[1],
        []
    )


def get_road_edges_multirow(
    road_mask,
    rows=None
):

    if rows is None:
        rows = DEFAULT_ROWS

    mask = _clean_mask(
        road_mask
    )

    h, _ = mask.shape

    left_points = []
    right_points = []

    for fraction in rows:

        y = int(
            fraction * h
        )

        result = _row_edges(
            mask,
            y
        )

        if result is None:
            continue

        left, right = result

        left_points.append(
            (left, y)
        )

        right_points.append(
            (right, y)
        )

    # Remove sudden jumps independently.
    left_points = _remove_jumps(
        left_points,
        max_jump=90
    )

    right_points = _remove_jumps(
        right_points,
        max_jump=90
    )

    flags = []

    if len(left_points) < 3:
        flags.append(
            EDGE_NOT_VISIBLE
        )

    if len(right_points) < 3:
        flags.append(
            EDGE_NOT_VISIBLE
        )

    left_line = _fit_boundary(
        left_points
    )

    right_line = _fit_boundary(
        right_points
    )

    return {
        "left_points": left_points,
        "right_points": right_points,
        "left_curve": left_line,
        "right_curve": right_line,
        "flags": flags,
    }


def check_edge_continuity(
    current_edge,
    previous_edge,
    max_jump_px
):

    if current_edge is None:
        return [EDGE_NOT_VISIBLE]

    if previous_edge is None:
        return []

    if abs(
        current_edge - previous_edge
    ) > max_jump_px:
        return [EDGE_JUMP]

    return []


def propagate_edge(
    current_edge,
    previous_edge,
    missing_frames,
    max_missing_frames=3
):

    if current_edge is not None:
        return current_edge, []

    if (
        previous_edge is not None
        and missing_frames <= max_missing_frames
    ):
        return previous_edge, [OCCLUSION]

    return None, [EDGE_NOT_VISIBLE]
