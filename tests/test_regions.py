import numpy as np

from gst_image.analysis.regions import classify_regions
from gst_image.models import RegionClassifierRecipe


def test_assisted_region_classifier_replays_deterministically():
    image = np.zeros((96, 128), np.uint8)
    image[:, :64] = 55
    image[:, 64:] = 205
    labels = np.zeros_like(image, np.uint8)
    labels[20:75, 15:25] = 1
    labels[20:75, 100:110] = 2
    recipe = RegionClassifierRecipe(
        sigma_max=4,
        n_estimators=20,
        max_depth=6,
        max_samples=1,
        random_seed=17,
    )
    first = classify_regions(image, labels, recipe)
    second = classify_regions(image, labels, recipe)
    assert np.array_equal(first.labels, second.labels)
    expected = np.ones_like(image)
    expected[:, 64:] = 2
    assert np.mean(first.labels == expected) > 0.98

