from collections import deque

import numpy as np


class TemporalWidthFilter:
    """
    Robust temporal filter for road-width measurements.

    Keeps a short history and returns the median width.
    Large sudden jumps are flagged instead of immediately
    becoming the accepted measurement.
    """

    def __init__(self, window_size=5, max_jump_m=1.5):
        self.window_size = window_size
        self.max_jump_m = max_jump_m
        self.history = deque(maxlen=window_size)

    def update(self, width_m):
        if width_m is None or not np.isfinite(width_m):
            return {
                "width_m": None,
                "accepted": False,
                "flag": "INVALID_WIDTH",
            }

        width_m = float(width_m)

        if self.history:
            previous = float(np.median(self.history))

            if abs(width_m - previous) > self.max_jump_m:
                return {
                    "width_m": previous,
                    "accepted": False,
                    "flag": "WIDTH_JUMP",
                }

        self.history.append(width_m)

        filtered_width = float(np.median(self.history))

        return {
            "width_m": filtered_width,
            "accepted": True,
            "flag": None,
        }

    def reset(self):
        self.history.clear()
