import os
from datetime import datetime
from typing import Any, cast
from unittest.mock import Mock, patch

import h5py
import numpy as np
from shapely import geometry as shpg

from vibe_core import file_downloader
from vibe_core.data import GEDIProduct
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.earthdata import EarthDataAPI

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "download_gedi_product.yaml")

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


def fake_download(_: str, h5_path: str, **kwargs: Any):
    beam_value = 0
    with h5py.File(h5_path, "w") as f:
        for b in BEAMS:
            beam_value = int(b.replace("BEAM", ""), 2)
            f.create_dataset(f"{b}/geolocation/lon_lowestmode", data=np.arange(NUM_POINTS))
            f.create_dataset(
                f"{b}/geolocation/lat_lowestmode", data=np.arange(NUM_POINTS) + NUM_POINTS
            )
            f.create_dataset(f"{b}/beam", data=beam_value * np.ones(NUM_POINTS))
            f.create_dataset(f"{b}/rh100", data=np.linspace(0, 1, NUM_POINTS) + beam_value)


@patch.object(file_downloader, "download_file")
@patch.object(EarthDataAPI, "query")
def test_op(query: Mock, download: Mock):
    query.return_value = [{"links": [{"href": "mock_link"}]}]
    download.side_effect = fake_download
    now = datetime.now()
    geom = shpg.box(0, 0, 1, 1)
    x = GEDIProduct(
        id="1",
        time_range=(now, now),
        geometry=shpg.mapping(geom),
        assets=[],
        product_name="fake_product",
        start_orbit=0,
        stop_orbit=0,
        processing_level="whatever",
    )
    op_tester = OpTester(CONFIG_PATH)
    test_token = "test-token"
    op_tester.update_parameters({"token": test_token})
    out = op_tester.run(gedi_product=x)
    query.assert_called_once_with(id=x.product_name)
    download.assert_called_once()
    # Make sure we used the token
    assert download.call_args.kwargs["headers"]["Authorization"] == f"Bearer {test_token}"
    assert "downloaded_product" in out
    dl_product = cast(GEDIProduct, out["downloaded_product"])
    assert dl_product.geometry == x.geometry
    assert dl_product.time_range == x.time_range
