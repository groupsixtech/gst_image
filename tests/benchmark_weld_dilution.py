"""Manual native panorama performance check; candidate union is NOT a reviewed weld."""

import argparse
import json
import platform
import time

from benchmark_large_image import peak_working_set_bytes

from gst_image.analysis.masks import suggest_specimen_mask
from gst_image.analysis.preprocess import to_gray
from gst_image.analysis.weld_dilution import (
    fit_surface_reference,
    measure_weld_dilution,
    segment_weld_envelope,
)
from gst_image.image_io import load_image
from gst_image.models import WeldDilutionRecipe
from gst_image.project import dependency_versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    start = time.perf_counter()
    image = load_image(args.image, color=True)
    domain = suggest_specimen_mask(to_gray(image))
    recipe = WeldDilutionRecipe()
    ready = time.perf_counter()
    candidates = segment_weld_envelope(image, domain, recipe).candidates
    segmented = time.perf_counter()
    envelope = candidates > 0
    del candidates, image
    reference = fit_surface_reference(
        [(0, domain.shape[0] / 2), (domain.shape[1] - 1, domain.shape[0] / 2)]
    )
    summary, lines = measure_weld_dilution(envelope, domain, reference)
    end = time.perf_counter()
    peak = peak_working_set_bytes()
    print(
        json.dumps(
            {
                "purpose": "Performance only: candidate union and synthetic mid-image reference",
                "image": args.image,
                "shape": domain.shape,
                "megapixels": domain.size / 1e6,
                "load_domain_seconds": ready - start,
                "segmentation_seconds": segmented - ready,
                "measurement_seconds": end - segmented,
                "total_seconds": end - start,
                "peak_working_set_gib": peak / 1024**3 if peak else None,
                "tie_lines": len(lines),
                "sample_counts": summary.sample_counts,
                "recipe": recipe.model_dump(mode="json"),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "dependencies": dependency_versions(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
