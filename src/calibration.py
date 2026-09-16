import json
from pathlib import Path

import cv2
import numpy as np


def calibrate_camera(
    image_paths,
    board_size=(9, 6),
    square_size_m=0.024,
):
    """
    Calibrate a camera using checkerboard images.

    board_size:
        Number of inner corners (columns, rows).

    square_size_m:
        Physical checkerboard square size in metres.

    Returns:
        camera_matrix
        distortion_coefficients
        rms_error
    """

    object_points = []
    image_points = []

    cols, rows = board_size

    object_template = np.zeros(
        (rows * cols, 3),
        dtype=np.float32,
    )

    object_template[:, :2] = np.mgrid[
        0:cols,
        0:rows,
    ].T.reshape(-1, 2)

    object_template *= square_size_m

    image_size = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            None,
        )

        if not found:
            continue

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )

        object_points.append(object_template.copy())
        image_points.append(corners)

    if not object_points:
        raise RuntimeError(
            "No checkerboard calibration images were detected."
        )

    rms_error, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    return {
        "camera_matrix": camera_matrix,
        "distortion_coefficients": distortion,
        "rms_error": float(rms_error),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "board_size": list(board_size),
        "square_size_m": float(square_size_m),
    }


def save_calibration(calibration, output_path):
    """Save calibration parameters as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "camera_matrix": np.asarray(
            calibration["camera_matrix"]
        ).tolist(),

        "distortion_coefficients": np.asarray(
            calibration["distortion_coefficients"]
        ).tolist(),

        "rms_error": calibration["rms_error"],
        "image_width": calibration["image_width"],
        "image_height": calibration["image_height"],
        "board_size": calibration["board_size"],
        "square_size_m": calibration["square_size_m"],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_calibration(path):
    """Load calibration parameters from JSON."""

    with open(path, "r") as f:
        data = json.load(f)

    data["camera_matrix"] = np.asarray(
        data["camera_matrix"],
        dtype=np.float64,
    )

    data["distortion_coefficients"] = np.asarray(
        data["distortion_coefficients"],
        dtype=np.float64,
    )

    return data
