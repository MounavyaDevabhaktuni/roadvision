import cv2
import numpy as np

from src.segmentation import segment_road
from src.boundaries import get_road_edges_multirow
from src.geometry import road_width_from_pixels
from src.calibration import load_calibration


IMAGE = "data/processed/frame_000000.jpg"
CALIBRATION = "models/calibration/camera.json"
OUTPUT = "data/processed/current_pipeline_check.jpg"


# Load image
image = cv2.imread(IMAGE)

if image is None:
    raise RuntimeError(
        f"Could not load {IMAGE}. Run frame extraction first."
    )

# Stage 3: segmentation
mask, seg_score = segment_road(image)

# Stage 4: boundaries
edges = get_road_edges_multirow(mask)

# Load provisional calibration
calibration = load_calibration(CALIBRATION)

camera_matrix = calibration["camera_matrix"]
distortion = calibration["distortion_coefficients"]


# Draw segmentation overlay
overlay = image.copy()
overlay[mask > 0] = (
    0.5 * overlay[mask > 0] +
    0.5 * np.array([0, 255, 0])
).astype(np.uint8)

result = cv2.addWeighted(
    image,
    0.6,
    overlay,
    0.4,
    0
)


# Draw detected boundary points
for x, y in edges["left_points"]:
    cv2.circle(result, (x, y), 7, (255, 0, 0), -1)

for x, y in edges["right_points"]:
    cv2.circle(result, (x, y), 7, (0, 0, 255), -1)


# Estimate width using the lowest detected row
if edges["left_points"] and edges["right_points"]:

    left_x, left_y = edges["left_points"][-1]
    right_x, right_y = edges["right_points"][-1]

    width_m = road_width_from_pixels(
        [left_x, left_y],
        [right_x, right_y],
        camera_matrix,
        distortion,
    )

    text = f"Estimated width: {width_m:.2f} m"

    cv2.putText(
        result,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )


cv2.putText(
    result,
    f"Segmentation score: {seg_score:.3f}",
    (30, 90),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2,
)

cv2.imwrite(OUTPUT, result)

print("Saved:", OUTPUT)
print("Segmentation score:", seg_score)
print("Left points:", edges["left_points"])
print("Right points:", edges["right_points"])

if edges["left_points"] and edges["right_points"]:
    print("Estimated width:", width_m, "metres")
