# Large-image benchmarks

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
