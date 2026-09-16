import torch
import cv2
import numpy as np

IMAGE = "/Users/mounavyadevabhaktuni/Desktop/comma2k19/comma_test_frame.png"

model = torch.hub.load(
    "hustvl/yolop",
    "yolop",
    pretrained=True
)

model.eval()

frame = cv2.imread(IMAGE)

if frame is None:
    raise RuntimeError("Could not read image")

h, w = frame.shape[:2]

# YOLOP expects 640 x 640
img = cv2.resize(frame, (640, 640))

# BGR -> RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# HWC -> CHW
img = img.transpose(2, 0, 1)

img = np.ascontiguousarray(img)

x = torch.from_numpy(img).float()
x = x.unsqueeze(0) / 255.0

with torch.no_grad():
    outputs = model(x)

# Drivable-area segmentation
da = outputs[1]

print("DA output:", da.shape)

# Get class probabilities
probabilities = torch.softmax(da, dim=1)

# Probability of drivable area
road_probability = probabilities[0, 1].cpu().numpy()

# Use a threshold instead of simply argmax
mask = (road_probability > 0.35).astype(np.uint8) * 255

# Resize back to original image
mask = cv2.resize(
    mask,
    (w, h),
    interpolation=cv2.INTER_NEAREST
)

# ------------------------------------------------
# CLEAN MASK
# ------------------------------------------------

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (11, 11)
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

# ------------------------------------------------
# GREEN OVERLAY
# ------------------------------------------------

overlay = frame.copy()

green = np.zeros_like(frame)
green[:, :, 1] = 255

road = mask > 0

overlay[road] = cv2.addWeighted(
    frame[road],
    0.45,
    green[road],
    0.55,
    0
)

# Put original and overlay side by side
result = np.hstack([
    frame,
    overlay
])

cv2.imwrite(
    "yolop_road_test.png",
    result
)

cv2.imwrite(
    "road_mask.png",
    mask
)

print("Saved yolop_road_test.png")
print("Saved road_mask.png")
print("Road pixels:", int(np.sum(road)))
