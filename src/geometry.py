import cv2
import numpy as np


def undistort_points(points, camera_matrix, distortion_coefficients):
    """Undistort image points."""

    points = np.asarray(points, dtype=np.float64)

    if points.size == 0:
        return points.reshape(-1, 2)

    points = points.reshape(-1, 1, 2)

    undistorted = cv2.undistortPoints(
        points,
        camera_matrix,
        distortion_coefficients,
        P=camera_matrix,
    )

    return undistorted.reshape(-1, 2)


def pixel_to_ground(
    points,
    camera_matrix,
    camera_height_m=1.5,
    pitch_deg=5.0,
):
    """
    Convert image pixels to approximate ground-plane coordinates.

    Returns:
        [X, Z]
        X = lateral distance in metres
        Z = forward distance in metres
    """

    points = np.asarray(points, dtype=np.float64)

    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    points = points.reshape(-1, 2)

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    pitch = np.deg2rad(pitch_deg)

    ground_points = []

    for u, v in points:

        x_camera = (u - cx) / fx
        y_camera = (v - cy) / fy

        ray = np.array(
            [
                x_camera,
                y_camera,
                1.0,
            ],
            dtype=np.float64,
        )

        rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [
                    0.0,
                    np.cos(pitch),
                    -np.sin(pitch),
                ],
                [
                    0.0,
                    np.sin(pitch),
                    np.cos(pitch),
                ],
            ]
        )

        ray_ground = rotation @ ray

        if ray_ground[1] <= 1e-6:
            ground_points.append([np.nan, np.nan])
            continue

        scale = camera_height_m / ray_ground[1]

        X = ray_ground[0] * scale
        Z = ray_ground[2] * scale

        ground_points.append([X, Z])

    return np.asarray(ground_points)


def ground_distance(point_a, point_b):
    """Calculate Euclidean distance between two ground points."""

    a = np.asarray(point_a, dtype=np.float64)
    b = np.asarray(point_b, dtype=np.float64)

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")

    return float(np.linalg.norm(a - b))


def road_width_from_pixels(
    left_pixel,
    right_pixel,
    camera_matrix,
    distortion_coefficients=None,
    camera_height_m=1.5,
    pitch_deg=5.0,
):
    """
    Calculate road width in metres from left/right road-edge pixels.
    """

    points = np.asarray(
        [left_pixel, right_pixel],
        dtype=np.float64,
    )

    if distortion_coefficients is not None:
        points = undistort_points(
            points,
            camera_matrix,
            distortion_coefficients,
        )

    ground_points = pixel_to_ground(
        points,
        camera_matrix,
        camera_height_m=camera_height_m,
        pitch_deg=pitch_deg,
    )

    width_m = ground_distance(
        ground_points[0],
        ground_points[1],
    )

    return float(width_m)
