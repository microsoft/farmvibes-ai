import os
from datetime import datetime, timezone
from typing import Any, Dict

from shapely import geometry as shpg

from vibe_core.data.sentinel import (
    Sentinel1RasterTileSequence,
    Sentinel2CloudMaskTileSequence,
    Sentinel2RasterTileSequence,
)
from vibe_dev.testing.op_tester import OpTester

CONFIG_PATH_NN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remove_clouds.yaml")

CONFIG_PATH_INTERP = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "remove_clouds_interpolation.yaml"
)


def test_remove_clouds_empty_sequence():
    polygon: Dict[str, Any] = shpg.mapping(shpg.box(0, 0, 1, 1))  # type: ignore
    start_date = datetime(year=2021, month=7, day=10, tzinfo=timezone.utc)
    end_date = datetime(year=2021, month=7, day=28, tzinfo=timezone.utc)
    s1 = Sentinel1RasterTileSequence(
        id="s1",
        time_range=(start_date, end_date),
        geometry=polygon,
        assets=[],
        product_name="",
        orbit_number=0,
        relative_orbit_number=0,
        orbit_direction="",
        platform="",
        extra_info={},
        sensor_mode="",
        polarisation_mode="",
        bands={},
        tile_id="",
        write_time_range=(start_date, end_date),
    )
    s2 = Sentinel2RasterTileSequence.clone_from(s1, id="s2", assets=[], processing_level="")
    cloud = Sentinel2CloudMaskTileSequence.clone_from(s2, id="cloud", assets=[], categories=[])

    nn_out = OpTester(CONFIG_PATH_NN).run(s1_products=s1, s2_products=s2, cloud_masks=cloud)
    assert not nn_out["spaceeye_sequence"].assets  # type: ignore
    interp_out = OpTester(CONFIG_PATH_INTERP).run(s2_products=s2, cloud_masks=cloud)
    assert not interp_out["spaceeye_sequence"].assets  # type: ignore
