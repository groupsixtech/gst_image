import os
from pathlib import Path

import pytest

from gst_image.image_io import load_preview


@pytest.mark.large
@pytest.mark.skipif(os.environ.get("RUN_LARGE_TESTS") != "1", reason="set RUN_LARGE_TESTS=1")
def test_all_sample_micrographs_decode_as_previews():
    images = sorted(Path("test/img").glob("*"))
    assert images
    for image in images:
        preview, full_size = load_preview(image, (1200, 800))
        assert preview.size
        assert full_size[0] >= preview.shape[1]
        assert full_size[1] >= preview.shape[0]

