import os
from datetime import datetime, timezone
from typing import Any, Dict

from shapely import geometry as shpg

from vibe_core.data.sentinel import SpaceEyeRasterSequence
from vibe_dev.testing.op_tester import OpTester

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "split_spaceeye_sequence.yaml"
)


def test_split_empty_sequence():
    polygon: Dict[str, Any] = shpg.mapping(shpg.box(0, 0, 1, 1))  # type: ignore
    start_date = datetime(year=2021, month=7, day=10, tzinfo=timezone.utc)
    end_date = datetime(year=2021, month=7, day=28, tzinfo=timezone.utc)
    seq = SpaceEyeRasterSequence(
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
        tile_id="",
        processing_level="",
        bands={},
        write_time_range=(start_date, end_date),
    )
    out = OpTester(CONFIG_PATH).run(sequences=[seq])
    assert not out["rasters"]
