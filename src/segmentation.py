import sys
from pathlib import Path

import cv2
import numpy as np
import torch


YOLOP_ROOT = Path.home() / ".cache/torch/hub/hustvl_yolop_main"

if str(YOLOP_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOP_ROOT))

from lib.utils import letterbox_for_img


_MODEL = None


def _load_model():
    global _MODEL

    if _MODEL is None:
        _MODEL = torch.hub.load(
            "hustvl/yolop",
            "yolop",
            pretrained=True
        )
        _MODEL.eval()

    return _MODEL


def _clean_mask(mask):
    """
    Post-process YOLOP drivable-area mask.
    """

    mask = (mask > 0).astype(np.uint8)

    height, width = mask.shape

    # Remove small isolated regions.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned = np.zeros_like(mask)

    min_area = int(height * width * 0.001)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label] = 1

    # Close small gaps inside the road.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Remove tiny remaining objects.
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        kernel_small
    )

    return cleaned.astype(np.uint8)


def segment_road(image):
    """
    Detect the drivable road area.

    Returns:
        road_mask:
            0 = not road
            1 = drivable road

        seg_score:
            fraction of image predicted as road
    """

    if image is None:
        raise ValueError("Input image is None")

    model = _load_model()

    original_height, original_width = image.shape[:2]

    # YOLOP preprocessing
    processed, ratio, pad = letterbox_for_img(
        image,
        new_shape=(640, 640),
        auto=True
    )

    # BGR -> RGB
    processed = processed[:, :, ::-1]

    # HWC -> CHW
    tensor = processed.transpose(2, 0, 1).copy()

    tensor = torch.from_numpy(tensor).float()
    tensor = tensor.unsqueeze(0) / 255.0

    # YOLOP inference
    with torch.no_grad():
        outputs = model(tensor)

    # Drivable-area output
    da_output = outputs[1]

    # Probability that each pixel belongs to the
    # drivable-area class.
    road_probability = torch.softmax(
        da_output,
        dim=1
    )[0, 1].cpu().numpy()

    # Use a moderate confidence threshold.
    mask = (road_probability >= 0.40).astype(np.uint8)

    # Remove letterbox padding
    pad_x, pad_y = pad

    left = int(round(pad_x))
    top = int(round(pad_y))

    mask_height, mask_width = mask.shape

    right = mask_width - int(round(pad_x))
    bottom = mask_height - int(round(pad_y))

    mask = mask[top:bottom, left:right]

    # Return to original frame size
    mask = cv2.resize(
        mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    # Stage 3 post-processing
    mask = _clean_mask(mask)

    # Segmentation score
    seg_score = float(mask.mean())

    return mask, seg_score
