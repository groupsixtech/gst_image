"""Geometric weld dilution, using native pixel cells and a straight surface reference."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_sauvola

from gst_image.analysis.preprocess import to_gray
from gst_image.models import (
    Calibration,
    Point,
    ProjectManifest,
    SegmentationLayer,
    SurfaceReference,
    WeldDilutionDraft,
    WeldDilutionRecipe,
    WeldDilutionRun,
    WeldDilutionSummary,
    WeldSampling,
    WeldStatistics,
    WeldTieLine,
)


def _check(cancelled):
    if cancelled and cancelled():
        raise InterruptedError("Weld dilution cancelled")


def fit_surface_reference(points, *, flipped: bool = False) -> SurfaceReference:
    """Orthogonal least-squares fit; endpoint order cannot reverse substrate side."""
    xy = np.asarray([(p.x, p.y) if isinstance(p, Point) else p for p in points], float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError("Supply at least two finite surface points")
    origin = xy.mean(axis=0)
    _, singular, vectors = np.linalg.svd(xy - origin, full_matrices=False)
    if singular[0] < 1e-8 or (len(singular) > 1 and singular[0] - singular[1] < 1e-8):
        raise ValueError("Surface points do not define a unique line")
    tangent = vectors[0]
    if tangent[0] < -1e-10 or (abs(tangent[0]) <= 1e-10 and tangent[1] < 0):
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]]) * (-1 if flipped else 1)
    return SurfaceReference(
        points=[Point(x=x, y=y) for x, y in xy],
        origin=Point(x=origin[0], y=origin[1]),
        tangent=Point(x=tangent[0], y=tangent[1]),
        normal=Point(x=normal[0], y=normal[1]),
        flipped=flipped,
        rms_residual_px=float(np.sqrt(np.mean(((xy - origin) @ normal) ** 2))),
        span_px=float(np.ptp((xy - origin) @ tangent)),
    )


def line_bounds(origin, vector, shape):
    """Parameter interval inside the image's pixel-cell rectangle, or None."""
    lo, hi = -np.inf, np.inf
    for value, direction, size in zip(origin, vector, reversed(shape)):
        if abs(direction) < 1e-12:
            if value < -0.5 or value > size - 0.5:
                return None
        else:
            a, b = sorted(((-0.5 - value) / direction, (size - 0.5 - value) / direction))
            lo, hi = max(lo, a), min(hi, b)
    return (float(lo), float(hi)) if hi > lo else None


def scaled_weld_recipe(recipe: WeldDilutionRecipe, scale: float) -> WeldDilutionRecipe:
    values = recipe.model_dump()
    values.update(
        gaussian_sigma_px=recipe.gaussian_sigma_px * scale,
        window_px=max(3, round(recipe.window_px * scale)),
        close_radius_px=round(recipe.close_radius_px * scale),
        open_radius_px=round(recipe.open_radius_px * scale),
        blur_strokes=[
            {"points": [{"x": p.x * scale, "y": p.y * scale} for p in stroke.points],
             "radius_px": stroke.radius_px * scale, "sigma_px": stroke.sigma_px * scale}
            for stroke in recipe.blur_strokes
        ],
    )
    return WeldDilutionRecipe(**values)


@dataclass
class WeldSegmentation:
    candidates: np.ndarray
    blurred: np.ndarray | None = None


def _gaussian_blur(gray, sigma):
    radius = math.ceil(4 * sigma)
    return (
        cv2.GaussianBlur(gray, (2 * radius + 1,) * 2, sigma, borderType=cv2.BORDER_REPLICATE)
        if radius
        else gray
    )


def _blur_radius(recipe):
    return math.ceil(4 * max(
        [recipe.gaussian_sigma_px] + [stroke.sigma_px for stroke in recipe.blur_strokes]
    ))


def _smooth_weld_channel(gray, recipe, x0, y0):
    """Local strokes replace smoothing strength, always sampling the original channel."""
    result = _gaussian_blur(gray, recipe.gaussian_sigma_px).copy()
    for stroke in recipe.blur_strokes:
        xy = np.asarray([(p.x, p.y) for p in stroke.points]) - [x0, y0]
        radius = stroke.radius_px
        if (np.any(xy.max(axis=0) + radius < 0)
                or np.any(xy.min(axis=0) - radius >= [gray.shape[1], gray.shape[0]])):
            continue
        # Source-coordinate capsule distances avoid tile-dependent line clipping/rasterization.
        mask = np.zeros(gray.shape, bool)
        points = stroke.points
        for a, b in zip(points, points[1:] or points):
            left = max(x0, math.floor(min(a.x, b.x) - radius))
            top = max(y0, math.floor(min(a.y, b.y) - radius))
            right = min(x0 + gray.shape[1], math.ceil(max(a.x, b.x) + radius) + 1)
            bottom = min(y0 + gray.shape[0], math.ceil(max(a.y, b.y) + radius) + 1)
            if right <= left or bottom <= top:
                continue
            ys, xs = np.ogrid[top:bottom, left:right]
            dx, dy = b.x - a.x, b.y - a.y
            length2 = dx * dx + dy * dy
            t = np.clip(((xs - a.x) * dx + (ys - a.y) * dy) / length2, 0, 1) \
                if length2 else 0
            selected = (xs - a.x - t * dx)**2 + (ys - a.y - t * dy)**2 <= radius**2
            mask[top-y0:bottom-y0, left-x0:right-x0] |= selected
        sigma = max(recipe.gaussian_sigma_px, stroke.sigma_px)
        local = _gaussian_blur(gray, sigma)
        np.copyto(result, local, where=mask)
    return result


def blurred_weld_preview(
    image, recipe: WeldDilutionRecipe, *, progress=None, cancelled=None, max_size=(2400, 1400),
):
    """Blur native pixels independently of thresholding, then reduce for display."""
    _check(cancelled)
    gray = to_gray(image, recipe.channel)
    height, width = gray.shape
    blurred = np.empty_like(gray)
    radius = _blur_radius(recipe)
    tile = recipe.tile_size_px
    count = math.ceil(height / tile) * math.ceil(width / tile)
    completed = 0
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            _check(cancelled)
            bottom, right = min(height, y + tile), min(width, x + tile)
            y0, x0 = max(0, y - radius), max(0, x - radius)
            local = _smooth_weld_channel(
                gray[y0:min(height, bottom + radius), x0:min(width, right + radius)],
                recipe, x0, y0,
            )
            blurred[y:bottom, x:right] = local[y-y0:bottom-y0, x-x0:right-x0]
            completed += 1
            if progress:
                progress(completed / count, "Previewing Gaussian blur — no segmentation")
    _check(cancelled)
    if max_size is None:
        return blurred
    scale = min(1.0, max_size[0] / width, max_size[1] / height)
    return cv2.resize(
        blurred,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def segment_weld_envelope(
    image,
    domain,
    recipe: WeldDilutionRecipe,
    *,
    progress=None,
    cancelled=None,
    return_blurred=False,
) -> WeldSegmentation:
    """Return selectable components, never automatically choose the largest object."""
    gray = to_gray(image, recipe.channel)
    domain = np.asarray(domain, bool)
    if domain.shape != gray.shape or not domain.any():
        raise ValueError("Weld analysis requires a nonempty matching domain")
    radius = _blur_radius(recipe)
    halo = radius + recipe.window_px // 2 + 2 * (recipe.close_radius_px + recipe.open_radius_px)
    binary = np.zeros(gray.shape, np.uint8)
    blurred = np.empty_like(gray) if return_blurred else None
    h, w = gray.shape
    tile = recipe.tile_size_px
    count = math.ceil(h / tile) * math.ceil(w / tile)
    completed = 0
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            _check(cancelled)
            bottom, right = min(h, y + tile), min(w, x + tile)
            y0, x0 = max(0, y - halo), max(0, x - halo)
            y1, x1 = min(h, bottom + halo), min(w, right + halo)
            local = gray[y0:y1, x0:x1]
            local = _smooth_weld_channel(local, recipe, x0, y0)
            core = np.s_[y - y0 : bottom - y0, x - x0 : right - x0]
            if blurred is not None:
                blurred[y:bottom, x:right] = local[core]
            if recipe.threshold_method == "adaptive_gaussian":
                mode = cv2.THRESH_BINARY_INV if recipe.polarity == "dark" else cv2.THRESH_BINARY
                selected = cv2.adaptiveThreshold(
                    local,
                    1,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    mode,
                    recipe.window_px,
                    recipe.adaptive_c,
                )
            else:
                threshold = threshold_sauvola(local, recipe.window_px, recipe.sauvola_k, r=128)
                selected = (
                    (local < threshold) if recipe.polarity == "dark" else (local > threshold)
                ).astype(np.uint8)
            for operation, r in (
                (cv2.MORPH_CLOSE, recipe.close_radius_px),
                (cv2.MORPH_OPEN, recipe.open_radius_px),
            ):
                if r:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
                    selected = cv2.morphologyEx(
                        selected,
                        operation,
                        kernel,
                        borderType=cv2.BORDER_REPLICATE,
                    )
            binary[y:bottom, x:right] = selected[core] & domain[y:bottom, x:right]
            completed += 1
            if progress:
                progress(0.8 * completed / count, "Segmenting weld candidates")
    _check(cancelled)
    binary = ndi.binary_fill_holes(binary) & domain
    _, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    _check(cancelled)
    if progress:
        progress(1, "Select the weld components and review the fusion boundary")
    return WeldSegmentation(labels, blurred)


def envelope_from_seeds(candidates, seeds, domain):
    ids = set()
    for p in seeds:
        x, y = math.floor(p.x + 0.5), math.floor(p.y + 0.5)
        if not (0 <= y < candidates.shape[0] and 0 <= x < candidates.shape[1]):
            raise ValueError("A component seed is outside the source image")
        label = int(candidates[y, x])
        if label == 0:
            raise ValueError("A saved component seed no longer hits a threshold candidate")
        ids.add(label)
    if not ids:
        raise ValueError("Select at least one weld component")
    return ndi.binary_fill_holes(np.isin(candidates, list(ids))) & np.asarray(domain, bool)


def _positive_fraction(distance, normal):
    """Exact area of a unit square on the positive side, using a uniform-sum CDF."""
    a, b = sorted(abs(float(v)) for v in normal)
    if a < 1e-12:
        return np.clip(distance / b + 0.5, 0, 1)
    t = distance + (a + b) / 2
    out = np.zeros_like(distance, dtype=float)
    out[t >= a + b] = 1
    middle = (t > 0) & (t < a + b)
    z = t[middle]
    out[middle] = (z**2 - np.maximum(z - a, 0) ** 2 - np.maximum(z - b, 0) ** 2) / (2 * a * b)
    return np.clip(out, 0, 1)


def divided_weld_mask(envelope, reference):
    """Display labels: 1 penetration (orange), 2 reinforcement (cyan)."""
    output = np.zeros(envelope.shape, np.uint8)
    n, o = reference.normal, reference.origin
    x = np.arange(envelope.shape[1]) - o.x
    for y in range(envelope.shape[0]):
        output[y] = np.where(n.x * x + n.y * (y - o.y) >= 0, 1, 2) * (envelope[y] > 0)
    return output


def _ray_intervals(origin, vector, envelope, domain):
    """Exact intersections with the union of closed unit pixel cells."""
    bounds = line_bounds(origin, vector, envelope.shape)
    if bounds is None:
        return [], []
    lo, hi = bounds
    crossings = [np.array([lo, hi])]
    for p, v, size in zip(origin, vector, reversed(envelope.shape)):
        if abs(v) > 1e-12:
            lower, upper = sorted((p + lo * v, p + hi * v))
            start = max(0, math.ceil(lower + 0.5))
            stop = min(size, math.floor(upper + 0.5))
            edges = (np.arange(start, stop + 1) - 0.5 - p) / v
            crossings.append(edges[(edges > lo + 1e-9) & (edges < hi - 1e-9)])
    ts = np.unique(np.concatenate(crossings))
    ts = ts[np.r_[True, np.diff(ts) > 1e-9]]
    mid = (ts[:-1] + ts[1:]) / 2
    xy = origin[:, None] + vector[:, None] * mid
    xs = np.clip(np.floor(xy[0] + 0.5).astype(int), 0, envelope.shape[1] - 1)
    ys = np.clip(np.floor(xy[1] + 0.5).astype(int), 0, envelope.shape[0] - 1)
    selected = envelope[ys, xs]
    valid_domain = domain[ys, xs]
    changes = np.diff(np.r_[False, selected, False].astype(np.int8))
    intervals, clipped = [], []
    for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        intervals.append((float(ts[first]), float(ts[last])))
        clipped.append(
            first == 0
            or last == len(selected)
            or not valid_domain[first - 1]
            or not valid_domain[last]
        )
    return intervals, clipped


def _statistics(values):
    if not values:
        return WeldStatistics()
    a = np.asarray(values, float)
    return WeldStatistics(
        count=len(a),
        mean=float(a.mean()),
        median=float(np.median(a)),
        std=float(a.std()),
        minimum=float(a.min()),
        maximum=float(a.max()),
        p5=float(np.percentile(a, 5)),
        p95=float(np.percentile(a, 95)),
    )


def measure_weld_dilution(
    envelope,
    domain,
    reference: SurfaceReference,
    calibration: Calibration | None = None,
    sampling: WeldSampling | None = None,
    *,
    progress=None,
    cancelled=None,
) -> tuple[WeldDilutionSummary, list[WeldTieLine]]:
    sampling = sampling or WeldSampling()
    domain = np.asarray(domain, bool)
    envelope = np.asarray(envelope, bool)
    if envelope.ndim != 2 or envelope.shape != domain.shape:
        raise ValueError("Weld envelope and domain must be matching two-dimensional masks")
    envelope = envelope & domain
    total = int(envelope.sum())
    if not total:
        raise ValueError("Weld envelope is empty")
    if calibration is None and (sampling.spacing_unit != "px" or sampling.weld_length_mm):
        raise ValueError("Physical spacing and volume estimates require calibration")
    scale = calibration.mm_per_pixel if calibration else None
    spacing = sampling.spacing
    if sampling.spacing_unit != "px":
        spacing *= (0.001 if sampling.spacing_unit == "µm" else 1) / scale
    o = np.array([reference.origin.x, reference.origin.y])
    n = np.array([reference.normal.x, reference.normal.y])
    t = np.array([reference.tangent.x, reference.tangent.y])
    limits = line_bounds(o, t, envelope.shape)
    if limits is None:
        raise ValueError("The surface reference does not intersect the image")
    span = limits[1] - limits[0]
    offset = min(spacing / 2, span / 2)
    count = max(1, math.ceil((span - offset) / spacing))
    if count > 1_000_000:
        raise ValueError("Tie-line spacing produces more than one million stations")
    base = 0.0
    for y in range(envelope.shape[0]):
        _check(cancelled)
        xs = np.flatnonzero(envelope[y])
        base += float(_positive_fraction(n[0] * (xs - o[0]) + n[1] * (y - o[1]), n).sum())
    partial = bool(np.any(envelope & ~ndi.binary_erosion(domain, border_value=0)))
    lines = []
    for index in range(count):
        _check(cancelled)
        # Half-step stations avoid sampling exactly on the outer image border.
        s = limits[0] + offset + index * spacing
        origin = o + s * t
        intervals, clipped = _ray_intervals(origin, n, envelope, domain)
        status = "no_weld"
        depth = height = thickness = ratio = None
        if intervals:
            contact = [i for i, (a, b) in enumerate(intervals) if a <= 1e-8 and b >= -1e-8]
            if any(clipped):
                status = "clipped"
            elif len(intervals) > 1:
                status = "multiple"
            elif not contact:
                status = "no_contact"
            else:
                status = "valid"
                a, b = intervals[contact[0]]
                depth, height = max(0.0, b), max(0.0, -a)
                thickness = depth + height
                ratio = 100 * depth / thickness if thickness else None
        lines.append(
            WeldTieLine(
                index=index,
                position_px=s - limits[0],
                x=origin[0],
                y=origin[1],
                status=status,
                depth_px=depth,
                height_px=height,
                thickness_px=thickness,
                local_ratio_percent=ratio,
            )
        )
        if progress and index % 100 == 0:
            progress(index / max(1, count), "Measuring perpendicular tie-lines")
    depths = [p.depth_px for p in lines if p.status == "valid"]
    heights = [p.height_px for p in lines if p.status == "valid"]
    areas = {
        "penetration_area_px2": base,
        "reinforcement_area_px2": total - base,
        "envelope_area_px2": float(total),
    }
    physical = {}
    if scale:
        physical = {key.replace("px2", "mm2"): value * scale**2 for key, value in areas.items()}
        if sampling.weld_length_mm:
            physical.update(
                {
                    key.replace("area_mm2", "volume_mm3"): value * sampling.weld_length_mm
                    for key, value in list(physical.items())
                }
            )
            physical["volume_assumption"] = (
                "Assuming constant cross-section along the entered length"
            )
    flags = ["partial_section"] if partial else []
    if not depths:
        flags.append("no_valid_tie_lines")
    summary = WeldDilutionSummary(
        **areas,
        **physical,
        dilution_percent=100 * base / total,
        depth_px=_statistics(depths),
        height_px=_statistics(heights),
        depth_mm=_statistics([v * scale for v in depths]) if scale else None,
        height_mm=_statistics([v * scale for v in heights]) if scale else None,
        sample_counts=dict(Counter(p.status for p in lines)),
        quality_flags=flags,
    )
    _check(cancelled)
    if progress:
        progress(1, "Weld dilution measured")
    return summary, lines


def append_dilution_run(
    manifest: ProjectManifest,
    masks: dict,
    draft: WeldDilutionDraft,
    envelope,
    domain,
    summary,
    tie_lines,
) -> WeldDilutionRun:
    """Snapshot masks and inputs; later draft editing cannot alter historical results."""
    layer = SegmentationLayer(
        name=draft.name,
        kind="weld_envelope",
        review_status="pending",
        scope_roi_ids=draft.scope_roi_ids,
    )
    domain_layer = SegmentationLayer(
        name=f"{draft.name} domain",
        kind="weld_domain",
        visible=False,
        scope_roi_ids=draft.scope_roi_ids,
        review_status="pending",
    )
    run = WeldDilutionRun(
        draft_id=draft.id,
        name=draft.name,
        reference=draft.reference.model_copy(deep=True),
        recipe=draft.recipe.model_copy(deep=True),
        sampling=draft.sampling.model_copy(deep=True),
        calibration=manifest.calibration.model_copy(deep=True) if manifest.calibration else None,
        scope_roi_ids=list(draft.scope_roi_ids),
        component_seeds=list(draft.component_seeds),
        envelope_layer_id=layer.id,
        domain_layer_id=domain_layer.id,
        layer_ids=[layer.id, domain_layer.id],
        source_sha256=manifest.source_sha256,
        source_revision=manifest.source_revision,
        summary=summary,
        tie_lines=tie_lines,
        manually_edited=draft.manually_edited,
    )
    for item, values in ((layer, envelope), (domain_layer, domain)):
        item.source_run_id = run.id
        manifest.layers.append(item)
        masks[item.id] = np.asarray(values, np.uint8).copy()
    manifest.dilution_runs.append(run)
    draft.last_run_id = run.id
    return run
