"""Area-fraction and stereological volume-fraction estimates."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from gst_image.models import FractionEntry, FractionResults, ProjectManifest


def compute_area_fractions(
    layers: Mapping[str, np.ndarray], analysis_mask: np.ndarray
) -> FractionResults:
    domain = np.asarray(analysis_mask, dtype=bool)
    denominator = int(np.count_nonzero(domain))
    if denominator == 0:
        raise ValueError("Analysis mask is empty")
    entries: list[FractionEntry] = []
    for name, layer in layers.items():
        values = np.asarray(layer)
        if values.shape != domain.shape:
            raise ValueError(f"Layer {name!r} does not match the analysis mask")
        count = int(np.count_nonzero(values.astype(bool) & domain))
        fraction = count / denominator
        entries.append(
            FractionEntry(
                layer=name,
                segmented_pixels=count,
                analyzed_pixels=denominator,
                area_fraction=fraction,
                estimated_volume_fraction_percent=100 * fraction,
            )
        )
    return FractionResults(entries=entries)


def compute_project_fractions(
    manifest: ProjectManifest, masks: Mapping[str, np.ndarray]
) -> FractionResults:
    """Expand binary and multi-class project layers using their matching analysis domains."""
    class_names = {item.id: item.name for item in manifest.classes}
    entries: list[FractionEntry] = []
    for layer in manifest.layers:
        if layer.kind == "domain" or layer.id not in masks:
            continue
        domain_layer = next(
            (
                item
                for item in manifest.layers
                if item.kind == "domain" and item.scope_roi_ids == layer.scope_roi_ids
            ),
            None,
        )
        domain = (
            masks[domain_layer.id].astype(bool)
            if domain_layer and domain_layer.id in masks
            else np.ones(masks[layer.id].shape, dtype=bool)
        )
        if layer.kind == "multiclass":
            expanded = {
                f"{layer.name}: {class_names.get(class_id, class_id)}": masks[layer.id] == value
                for value, class_id in layer.class_value_map.items()
            }
        else:
            expanded = {layer.name: masks[layer.id] > 0}
        entries.extend(compute_area_fractions(expanded, domain).entries)
    return FractionResults(entries=entries)
