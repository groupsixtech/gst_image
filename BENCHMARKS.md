# Large-image benchmarks

## Weld dilution benchmark (2026-10-07)

`python tests/benchmark_weld_dilution.py "test/img/1199_1_stitch.jpg"`
processed the 25,574 × 6,154 panorama (157.38 MP) in **42.95 seconds**, with a
**1.966 GiB** peak process working set. Image/domain preparation took 8.16 s,
adaptive segmentation 13.83 s, and area/tie-line measurement 20.96 s. The run used
the default dilution recipe (sigma 10 px, adaptive Gaussian window 201 px, C=2,
closing radius 3 px, opening off, 2,048 px tiles) and 25,574 normal stations.

This is a capacity check using the union of threshold candidates and a synthetic
mid-image horizon, **not an operator-reviewed weld analysis or accuracy claim**.
Its stations were all flagged (13,863 clipped; 11,711 multiple intersections), as
expected for that deliberately unreviewed candidate union.

Environment: Windows 11 build 26200, Python 3.14.7, Intel64 Family 6 Model 186
Stepping 3; NumPy 2.5.2, OpenCV 4.14.0.94, SciPy 1.18.1, scikit-image 0.26.0,
PySide6 6.11.2, Pydantic 2.13.5. The benchmark prints full dependency metadata.

Run the manual benchmark from the repository root:

```powershell
python tests/benchmark_large_image.py "test/img/1199_1_stitch.jpg"
```

This reports wall time, process peak working set, image size, particle count, area fraction,
and the complete recipe. It performs full-resolution illumination correction, tiled Sauvola
thresholding, connected-component filtering, and measurements. Watershed is disabled in the
baseline because its runtime depends strongly on the number and topology of candidate clusters;
benchmark it separately for a production recipe when required.

## Verified environment

Results are recorded after running the benchmark, not estimated. Hardware, application version,
Python version, and dependency pins should accompany any result used for capacity planning.

Baseline recorded 2026-09-10:

- GST Image 0.1.0 and Python 3.14.7 on Windows 11 build 26200.
- Intel64 Family 6 Model 186 Stepping 3, 12 logical processors.
- Dependency versions are recorded in `requirements-lock.txt`.
- Input: `test/img/1199_1_stitch.jpg`, 25,574 x 6,154 pixels (157.38 MP).
- Two consecutive wall times: 86.58 and 86.81 seconds.
- Peak process working set: 3.595 GiB on the instrumented run.
- Output: 24,792 candidates and 12.378% area fraction with the baseline recipe.
- Recipe: rolling-ball radius 151 px; Sauvola window 101 px and k 0.2; 2,048 px
  tiles; 1 px opening/closing; 100-200,000 px^2 area bounds; dark polarity; watershed off.

The result demonstrates the first-release target of less than 4 GiB for the largest supplied
image on this environment. The candidate count and fraction are not accuracy claims: the recipe
has not yet been tuned against metallurgist-approved masks for this micrograph.
