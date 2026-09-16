import cv2
import numpy as np

from src.segmentation import segment_road
from src.boundaries import get_road_edges_multirow


video = "data/raw/comma2k19/video.hevc"

cap = cv2.VideoCapture(video)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read video")

mask, score = segment_road(frame)

result = get_road_edges_multirow(mask)

visual = frame.copy()


# Draw detected points
for x, y in result["left_points"]:
    cv2.circle(
        visual,
        (x, y),
        6,
        (255, 0, 0),
        -1
    )

for x, y in result["right_points"]:
    cv2.circle(
        visual,
        (x, y),
        6,
        (0, 0, 255),
        -1
    )


def draw_line(
    visual,
    line,
    points,
    color
):

    if line is None or len(points) < 3:
        return

    a, b = line

    ys = np.array(
        [p[1] for p in points],
        dtype=float
    )

    y1 = int(ys.min())
    y2 = int(ys.max())

    x1 = int(a * y1 + b)
    x2 = int(a * y2 + b)

    cv2.line(
        visual,
        (x1, y1),
        (x2, y2),
        color,
        4
    )


draw_line(
    visual,
    result["left_curve"],
    result["left_points"],
    (255, 0, 0)
)

draw_line(
    visual,
    result["right_curve"],
    result["right_points"],
    (0, 0, 255)
)


cv2.imwrite(
    "data/processed/boundaries_check.jpg",
    visual
)

print("Saved: data/processed/boundaries_check.jpg")
print("Segmentation score:", score)
print("Left points:", result["left_points"])
print("Right points:", result["right_points"])
print("Left line:", result["left_curve"])
print("Right line:", result["right_curve"])
print("Flags:", result["flags"])
