from datetime import datetime
from math import cos, radians, sin

from pyproj import Transformer

from app.schemas import FrameRecord
from src.gps import load_gps_track, interpolate_gps
from src.video import sample_frames


def _interpolate_value(a, b, ratio):
    """Linearly interpolate an optional numeric value."""

    if a is None:
        return b

    if b is None:
        return a

    return a + ratio * (b - a)


def _find_gps_bracket(gps_track, timestamp_s):
    """
    Find the GPS points immediately before and after a video timestamp.
    """

    if not gps_track:
        raise ValueError("GPS track is empty.")

    target = gps_track[0]["timestamp"].timestamp() + timestamp_s

    if target <= gps_track[0]["timestamp"].timestamp():
        return gps_track[0], gps_track[0], 0.0

    if target >= gps_track[-1]["timestamp"].timestamp():
        return gps_track[-1], gps_track[-1], 0.0

    for before, after in zip(gps_track, gps_track[1:]):
        t0 = before["timestamp"].timestamp()
        t1 = after["timestamp"].timestamp()

        if t0 <= target <= t1:
            if t1 == t0:
                ratio = 0.0
            else:
                ratio = (target - t0) / (t1 - t0)

            return before, after, ratio

    raise RuntimeError("Could not find GPS interpolation bracket.")


def _latlon_to_enu(lat, lon, alt, origin):
    """Convert WGS84 coordinates to local East-North-Up coordinates."""

    lat0, lon0, alt0 = origin

    to_ecef = Transformer.from_crs(
        "EPSG:4979",
        "EPSG:4978",
        always_xy=True,
    )

    x, y, z = to_ecef.transform(lon, lat, alt)
    x0, y0, z0 = to_ecef.transform(lon0, lat0, alt0)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    lat0_rad = radians(lat0)
    lon0_rad = radians(lon0)

    east = (
        -sin(lon0_rad) * dx
        + cos(lon0_rad) * dy
    )

    north = (
        -sin(lat0_rad) * cos(lon0_rad) * dx
        - sin(lat0_rad) * sin(lon0_rad) * dy
        + cos(lat0_rad) * dz
    )

    up = (
        cos(lat0_rad) * cos(lon0_rad) * dx
        + cos(lat0_rad) * sin(lon0_rad) * dy
        + sin(lat0_rad) * dz
    )

    return east, north, up


def _enu_to_latlon(east, north, up, origin):
    """Convert local East-North-Up coordinates back to WGS84."""

    lat0, lon0, alt0 = origin

    lat0_rad = radians(lat0)
    lon0_rad = radians(lon0)

    dx = (
        -sin(lon0_rad) * east
        - sin(lat0_rad) * cos(lon0_rad) * north
        + cos(lat0_rad) * cos(lon0_rad) * up
    )

    dy = (
        cos(lon0_rad) * east
        - sin(lat0_rad) * sin(lon0_rad) * north
        + cos(lat0_rad) * sin(lon0_rad) * up
    )

    dz = (
        cos(lat0_rad) * north
        + sin(lat0_rad) * up
    )

    to_ecef = Transformer.from_crs(
        "EPSG:4978",
        "EPSG:4979",
        always_xy=True,
    )

    to_ecef_x = Transformer.from_crs(
        "EPSG:4979",
        "EPSG:4978",
        always_xy=True,
    )

    x0, y0, z0 = to_ecef_x.transform(
        lon0,
        lat0,
        alt0,
    )

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    lon, lat, _alt = to_ecef.transform(
        x,
        y,
        z,
        direction="INVERSE",
    )

    return lat, lon


def _interpolate_gps_metric(gps_track, timestamp_s):
    """
    Interpolate latitude/longitude in a local ENU metric coordinate system.
    """

    before, after, ratio = _find_gps_bracket(
        gps_track,
        timestamp_s,
    )

    origin = (
        gps_track[0]["latitude"],
        gps_track[0]["longitude"],
        gps_track[0]["altitude_m"] or 0.0,
    )

    before_alt = before["altitude_m"] or 0.0
    after_alt = after["altitude_m"] or 0.0

    e0, n0, u0 = _latlon_to_enu(
        before["latitude"],
        before["longitude"],
        before_alt,
        origin,
    )

    e1, n1, u1 = _latlon_to_enu(
        after["latitude"],
        after["longitude"],
        after_alt,
        origin,
    )

    east = e0 + ratio * (e1 - e0)
    north = n0 + ratio * (n1 - n0)
    up = u0 + ratio * (u1 - u0)

    latitude, longitude = _enu_to_latlon(
        east,
        north,
        up,
        origin,
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "speed_mps": _interpolate_value(
            before["speed_mps"],
            after["speed_mps"],
            ratio,
        ),
        "altitude_m": _interpolate_value(
            before["altitude_m"],
            after["altitude_m"],
            ratio,
        ),
    }


def build_frame_records(
    video_path,
    gps_path=None,
    step_s=1.0,
    interpolate_in_metric=False,
):
    """
    Build synchronized FrameRecord objects.

    Each video frame receives the GPS position corresponding
    to its video-relative timestamp.
    """

    frames = sample_frames(
        video_path,
        step_s=step_s,
    )

    gps_track = load_gps_track(
        video_path,
        gps_path,
    )

    records = []

    for frame in frames:

        if interpolate_in_metric:
            gps = _interpolate_gps_metric(
                gps_track,
                frame["timestamp_s"],
            )
        else:
            gps = interpolate_gps(
                gps_track,
                frame["timestamp_s"],
            )

        records.append(
            FrameRecord(
                frame_id=frame["frame_id"],
                timestamp_s=frame["timestamp_s"],
                image_path=frame["image_path"],
                latitude=gps["latitude"],
                longitude=gps["longitude"],
                speed_mps=gps["speed_mps"],
                altitude_m=gps["altitude_m"],
            )
        )

    return records
