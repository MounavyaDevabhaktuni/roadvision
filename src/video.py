import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _get_video_info(video_path):
    """Get basic video information."""

    command = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    info = json.loads(result.stdout)

    video_stream = next(
        stream
        for stream in info["streams"]
        if stream["codec_type"] == "video"
    )

    fps_text = video_stream.get("avg_frame_rate", "0/1")
    fps_num, fps_den = map(int, fps_text.split("/"))

    fps = fps_num / fps_den if fps_den else 0.0

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    duration_text = info.get("format", {}).get("duration")

    duration = (
        float(duration_text)
        if duration_text is not None
        else None
    )

    return {
        "duration_s": duration,
        "fps": fps,
        "width": width,
        "height": height,
    }


def _load_comma_frame_times(video_path):
    """
    Load Comma2k19 frame timestamps if available.

    Expected layout:

        data/raw/comma2k19/
        ├── video.hevc
        └── global_pose/
            └── frame_times
    """

    video_path = Path(video_path)

    frame_times_path = (
        video_path.parent
        / "global_pose"
        / "frame_times"
    )

    if not frame_times_path.exists():
        return None

    return np.load(
        frame_times_path,
        allow_pickle=False,
    )


def sample_frames(video_path, step_s=1.0):
    """
    Sample frames approximately every step_s seconds.

    Supports normal videos and raw Comma2k19 HEVC streams.
    """

    video_path = Path(video_path)

    output_dir = Path("data/processed")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    info = _get_video_info(video_path)

    # --------------------------------------------------
    # Comma2k19 path
    # --------------------------------------------------

    frame_times = _load_comma_frame_times(video_path)

    if frame_times is not None:

        relative_times = (
            frame_times - frame_times[0]
        )

        target_times = np.arange(
            0.0,
            float(relative_times[-1]) + 1e-6,
            step_s,
        )

        selected_indices = []

        for target in target_times:

            index = int(
                np.argmin(
                    np.abs(relative_times - target)
                )
            )

            selected_indices.append(index)

        selected_indices = sorted(
            set(selected_indices)
        )

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        records = []

        for frame_id, frame_index in enumerate(
            selected_indices
        ):

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

            success, frame = cap.read()

            if not success:
                continue

            timestamp_s = float(
                relative_times[frame_index]
            )

            output_path = (
                output_dir
                / f"frame_{frame_id:06d}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                frame,
            )

            records.append(
                {
                    "frame_id": frame_id,
                    "timestamp_s": timestamp_s,
                    "image_path": str(output_path),
                }
            )

        cap.release()

        return records

    # --------------------------------------------------
    # Normal video path
    # --------------------------------------------------

    duration = info["duration_s"]

    if duration is None:
        raise RuntimeError(
            "Video duration is unavailable and no "
            "Comma2k19 frame_times file was found."
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    records = []

    timestamp_s = 0.0
    frame_id = 0

    while timestamp_s < duration:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp_s * 1000,
        )

        success, frame = cap.read()

        if not success:
            break

        output_path = (
            output_dir
            / f"frame_{frame_id:06d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame,
        )

        records.append(
            {
                "frame_id": frame_id,
                "timestamp_s": timestamp_s,
                "image_path": str(output_path),
            }
        )

        frame_id += 1
        timestamp_s += step_s

    cap.release()

    return records
