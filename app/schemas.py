from dataclasses import dataclass


@dataclass
class FrameRecord:
    frame_id: int
    timestamp_s: float
    image_path: str

    latitude: float | None
    longitude: float | None
    speed_mps: float | None
    altitude_m: float | None


@dataclass
class Measurement:
    timestamp_s: float
    latitude: float
    longitude: float

    width_m: float
    confidence: float
    flags: list[str]
    frame_path: str
