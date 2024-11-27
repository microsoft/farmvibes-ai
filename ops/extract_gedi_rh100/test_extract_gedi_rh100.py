# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from pathlib import Path
from typing import cast

import geopandas as gpd
import h5py
import numpy as np
import pytest
from shapely import geometry as shpg

from vibe_core.data import AssetVibe, DataVibe, GEDIProduct, GeometryCollection
from vibe_dev.testing.op_tester import OpTester

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "extract_gedi_rh100.yaml")

NUM_POINTS = 10
BEAMS = [
    "BEAM0000",
    "BEAM0001",
    "BEAM0010",
    "BEAM0011",
    "BEAM0101",
    "BEAM0110",
    "BEAM1000",
    "BEAM1011",
]
L2B = "GEDI02_B.002"


@pytest.fixture
def fake_asset(tmp_path: Path):
    beam_value = 0
    filepath = os.path.join(tmp_path.absolute(), "fake.h5")
    with h5py.File(filepath, "w") as f:
        for b in BEAMS:
            beam_value = int(b.replace("BEAM", ""), 2)
            f.create_dataset(f"{b}/geolocation/lon_lowestmode", data=np.linspace(0, 2, NUM_POINTS))
            f.create_dataset(f"{b}/geolocation/lat_lowestmode", data=np.linspace(0, 2, NUM_POINTS))
            f.create_dataset(f"{b}/beam", data=beam_value * np.ones(NUM_POINTS))
            f.create_dataset(f"{b}/rh100", data=np.linspace(0, 1, NUM_POINTS) + beam_value)
            fake_qual = np.ones(NUM_POINTS)
            fake_qual[0] = 0
            f.create_dataset(f"{b}/l2b_quality_flag", data=fake_qual)
    return filepath


@pytest.mark.parametrize("check_quality", (True, False))
def test_op(check_quality: bool, fake_asset: str):
    now = datetime.now()
    x = GEDIProduct(
        id="1",
        time_range=(now, now),
        geometry=shpg.mapping(shpg.box(0, 0, 2, 2)),
        product_name="fake_product",
        start_orbit=0,
        stop_orbit=0,
        processing_level=L2B,
        assets=[AssetVibe(reference=fake_asset, type="application/x-hdf5", id="fake-id")],
    )
    geom = shpg.box(-1, -1, 1, 1)
    roi = DataVibe(id="2", time_range=(now, now), geometry=shpg.mapping(geom), assets=[])
    op_tester = OpTester(CONFIG_PATH)
    op_tester.update_parameters({"check_quality": check_quality})
    out = op_tester.run(gedi_product=x, roi=roi)
    assert "rh100" in out
    rh100 = cast(GeometryCollection, out["rh100"])
    assert rh100.geometry == roi.geometry
    assert rh100.time_range == x.time_range

    df = gpd.read_file(rh100.assets[0].url)
    quality_offset = int(check_quality)
    num_points = NUM_POINTS // 2 - quality_offset
    assert df.shape[0] == len(BEAMS) * num_points
    assert all(isinstance(g, shpg.Point) for g in df.geometry)
    assert np.allclose(
        df["rh100"],  # type: ignore
        np.concatenate(
            [
                np.linspace(0, 1, NUM_POINTS)[quality_offset : num_points + quality_offset]
                + int(b.replace("BEAM", ""), 2)
                for b in BEAMS
            ]
        ),
    )

    # Op breaks with wrong processing level
    x.processing_level = "invalid"
    with pytest.raises(ValueError):
        op_tester.run(gedi_product=x, roi=roi)
