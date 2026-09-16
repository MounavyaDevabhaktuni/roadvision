import json
import subprocess
from pathlib import Path

import cv2


def _get_video_info(video_path):
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

    duration = float(info["format"]["duration"])

    fps_num, fps_den = map(
        int,
        video_stream["avg_frame_rate"].split("/"),
    )

    fps = fps_num / fps_den

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    return {
        "duration_s": duration,
        "fps": fps,
        "width": width,
        "height": height,
    }


def fix_orientation(frame, timestamp_s):
    if timestamp_s < 20:
        return cv2.rotate(frame, cv2.ROTATE_180)

    return frame


def sample_frames(video_path, step_s=1.0):
    video_path = Path(video_path)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    info = _get_video_info(video_path)
    duration = info["duration_s"]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

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

        frame = fix_orientation(frame, timestamp_s)

        output_path = output_dir / f"frame_{frame_id:06d}.jpg"

        cv2.imwrite(str(output_path), frame)

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
