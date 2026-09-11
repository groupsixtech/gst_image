"""Manual full-resolution benchmark; not collected by pytest."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np

from gst_image.analysis import build_analysis_mask, segment_particles, suggest_specimen_mask
from gst_image.image_io import load_image
from gst_image.models import SegmentationRecipe


def peak_working_set_bytes() -> int | None:
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        handle = get_current_process()
        success = get_process_memory_info(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize) if success else None
    try:
        import resource

        maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(maximum * (1024 if os.name != "darwin" else 1))
    except (ImportError, AttributeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--tile-size", type=int, default=2048)
    args = parser.parse_args()
    started = time.perf_counter()
    image = load_image(args.image)
    domain = build_analysis_mask(image.shape, specimen_mask=suggest_specimen_mask(image))
    recipe = SegmentationRecipe(
        min_particle_area_px=100,
        max_particle_area_px=200_000,
        split_touching=False,
        tile_size_px=args.tile_size,
    )
    result = segment_particles(image, domain, recipe)
    elapsed = time.perf_counter() - started
    peak = peak_working_set_bytes()
    print(
        json.dumps(
            {
                "image": str(args.image),
                "shape": list(image.shape),
                "megapixels": float(np.prod(image.shape) / 1_000_000),
                "seconds": elapsed,
                "peak_working_set_gib": peak / 1024**3 if peak is not None else None,
                "particle_count": result.summary["particle_count"],
                "area_fraction": result.summary["area_fraction"],
                "recipe": recipe.model_dump(mode="json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
