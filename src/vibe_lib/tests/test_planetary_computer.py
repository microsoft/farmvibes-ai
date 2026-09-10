# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from vibe_lib.planetary_computer import LandsatCollection


def test_landsat_collection_excludes_non_raster_qa_asset():
    assert "qa" not in LandsatCollection.asset_keys
    assert "qa_pixel" in LandsatCollection.asset_keys
    assert "qa_radsat" in LandsatCollection.asset_keys
