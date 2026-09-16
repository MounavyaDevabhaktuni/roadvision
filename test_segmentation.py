import cv2
from src.segmentation import segment_road

video = "data/raw/comma2k19/video.hevc"

cap = cv2.VideoCapture(video)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read video")

mask, score = segment_road(frame)

# Visualization ONLY.
# Stage 4 will use the raw mask directly.
overlay = frame.copy()
overlay[mask == 1] = (0, 255, 0)

result = cv2.addWeighted(
    frame,
    0.6,
    overlay,
    0.4,
    0
)

cv2.imwrite(
    "data/processed/segmentation_check.jpg",
    result
)

print("Saved: data/processed/segmentation_check.jpg")
print("Segmentation score:", score)
