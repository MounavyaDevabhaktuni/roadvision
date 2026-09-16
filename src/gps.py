import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


GPS_ERROR = (
    "No GPS track detected. Provide a companion GPX/CSV or a camera "
    "profile that exposes the telemetry track."
)


def _parse_time(time_text):
    """Convert an ISO GPS timestamp into a timezone-aware datetime."""
    if not time_text:
        return None

    time_text = time_text.strip()

    if time_text.endswith("Z"):
        time_text = time_text[:-1] + "+00:00"

    return datetime.fromisoformat(time_text).astimezone(timezone.utc)


def _load_gpx(gpx_path):
    """Load GPS points from a GPX file."""
    root = ET.parse(gpx_path).getroot()

    points = []

    # GPX files use XML namespaces, so we match elements by their local name.
    for trkpt in root.iter():
        if trkpt.tag.split("}")[-1] != "trkpt":
            continue

        lat = trkpt.get("lat")
        lon = trkpt.get("lon")

        time_element = None
        ele_element = None
        speed_element = None

        for child in trkpt.iter():
            tag = child.tag.split("}")[-1]

            if tag == "time":
                time_element = child
            elif tag == "ele":
                ele_element = child
            elif tag == "speed":
                speed_element = child

        if lat is None or lon is None or time_element is None:
            continue

        point = {
            "timestamp": _parse_time(time_element.text),
            "latitude": float(lat),
            "longitude": float(lon),
            "altitude_m": (
                float(ele_element.text)
                if ele_element is not None and ele_element.text
                else None
            ),
            "speed_mps": (
                float(speed_element.text)
                if speed_element is not None and speed_element.text
                else None
            ),
        }

        points.append(point)

    return points


def _load_csv(csv_path):
    """Load GPS points from a simple CSV file."""
    points = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Accept common column-name variations.
            timestamp = (
                row.get("timestamp")
                or row.get("time")
                or row.get("datetime")
            )

            lat = row.get("latitude") or row.get("lat")
            lon = row.get("longitude") or row.get("lon")

            if not timestamp or not lat or not lon:
                continue

            points.append(
                {
                    "timestamp": _parse_time(timestamp),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "altitude_m": (
                        float(row["altitude_m"])
                        if row.get("altitude_m")
                        else None
                    ),
                    "speed_mps": (
                        float(row["speed_mps"])
                        if row.get("speed_mps")
                        else None
                    ),
                }
            )

    return points


def load_gps_track(video_path, gps_path=None):
    """
    Load a GPS track.

    First checks for a companion GPS file when supplied.
    Supports GPX and CSV.

    The video_path argument is kept because later we can add
    embedded camera telemetry support without changing the API.
    """

    # Currently, our public test dataset uses a companion GPX file.
    if gps_path is not None:
        gps_path = Path(gps_path)

        if gps_path.exists():
            suffix = gps_path.suffix.lower()

            if suffix == ".gpx":
                points = _load_gpx(gps_path)
            elif suffix == ".csv":
                points = _load_csv(gps_path)
            else:
                points = []

            if points:
                return points

    # Embedded telemetry support will be added for supported
    # camera formats/profiles.
    #
    # We deliberately do NOT fabricate GPS coordinates.
    raise RuntimeError(GPS_ERROR)


def interpolate_gps(gps_track, timestamp_s):
    """
    Interpolate GPS information at a video-relative timestamp.

    timestamp_s:
        Seconds from the beginning of the video.

    Returns:
        latitude
        longitude
        speed_mps
        altitude_m
    """

    if not gps_track:
        raise ValueError("GPS track is empty.")

    # Convert video-relative time into an absolute GPS timestamp.
    gps_start = gps_track[0]["timestamp"]

    target_time = gps_start.timestamp() + float(timestamp_s)

    # Handle before/after the GPS track.
    if target_time <= gps_track[0]["timestamp"].timestamp():
        point = gps_track[0]
        return {
            "latitude": point["latitude"],
            "longitude": point["longitude"],
            "speed_mps": point["speed_mps"],
            "altitude_m": point["altitude_m"],
        }

    if target_time >= gps_track[-1]["timestamp"].timestamp():
        point = gps_track[-1]
        return {
            "latitude": point["latitude"],
            "longitude": point["longitude"],
            "speed_mps": point["speed_mps"],
            "altitude_m": point["altitude_m"],
        }

    # Find the two GPS points surrounding the requested timestamp.
    for before, after in zip(gps_track, gps_track[1:]):
        t0 = before["timestamp"].timestamp()
        t1 = after["timestamp"].timestamp()

        if t0 <= target_time <= t1:
            if t1 == t0:
                ratio = 0.0
            else:
                ratio = (target_time - t0) / (t1 - t0)

            latitude = (
                before["latitude"]
                + ratio * (after["latitude"] - before["latitude"])
            )

            longitude = (
                before["longitude"]
                + ratio * (after["longitude"] - before["longitude"])
            )

            def interpolate_optional(key):
                a = before[key]
                b = after[key]

                if a is None:
                    return b

                if b is None:
                    return a

                return a + ratio * (b - a)

            return {
                "latitude": latitude,
                "longitude": longitude,
                "speed_mps": interpolate_optional("speed_mps"),
                "altitude_m": interpolate_optional("altitude_m"),
            }

    raise RuntimeError("Could not interpolate GPS timestamp.")
