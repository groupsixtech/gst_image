"""Calibration and line measurement helpers."""

from __future__ import annotations

import math

from gst_image.models import Calibration, Measurement, Point


def create_measurement(
    name: str,
    start: tuple[float, float],
    end: tuple[float, float],
    calibration: Calibration | None = None,
) -> Measurement:
    length_px = math.dist(start, end)
    return Measurement(
        name=name,
        start=Point(x=start[0], y=start[1]),
        end=Point(x=end[0], y=end[1]),
        length_px=length_px,
        length_mm=length_px * calibration.mm_per_pixel if calibration else None,
    )

