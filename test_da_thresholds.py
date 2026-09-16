import sys
from pathlib import Path

import cv2
import numpy as np
import torch

YOLOP_ROOT = Path.home() / ".cache/torch/hub/hustvl_yolop_main"

if str(YOLOP_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOP_ROOT))

from lib.utils import letterbox_for_img


model = torch.hub.load(
    "hustvl/yolop",
    "yolop",
    pretrained=True
)
model.eval()


video = "data/raw/comma2k19/video.hevc"

cap = cv2.VideoCapture(video)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read video")


h, w = frame.shape[:2]

processed, ratio, pad = letterbox_for_img(
    frame,
    new_shape=(640, 640),
    auto=True
)

processed = processed[:, :, ::-1]
tensor = processed.transpose(2, 0, 1).copy()
tensor = torch.from_numpy(tensor).float()
tensor = tensor.unsqueeze(0) / 255.0


with torch.no_grad():
    outputs = model(tensor)

da_output = outputs[1]

# Convert logits to probability of drivable area
probability = torch.softmax(da_output, dim=1)[0, 1].cpu().numpy()


# Remove padding from probability map
pad_x, pad_y = pad

left = int(round(pad_x))
top = int(round(pad_y))

ph, pw = probability.shape

right = pw - int(round(pad_x))
bottom = ph - int(round(pad_y))

probability = probability[top:bottom, left:right]

probability = cv2.resize(
    probability,
    (w, h),
    interpolation=cv2.INTER_LINEAR
)


for threshold in [0.20, 0.30, 0.40]:

    mask = (probability >= threshold).astype(np.uint8)

    overlay = frame.copy()
    overlay[mask == 1] = (0, 255, 0)

    result = cv2.addWeighted(
        frame,
        0.6,
        overlay,
        0.4,
        0
    )

    filename = f"data/processed/da_threshold_{int(threshold * 100)}.jpg"

    cv2.imwrite(filename, result)

    print(
        f"Threshold {threshold}: "
        f"road coverage = {mask.mean():.3f} "
        f"saved = {filename}"
    )
