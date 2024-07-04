import os
from datetime import datetime, timezone
from typing import List, cast

from shapely import affinity as shpa
from shapely import geometry as shpg

from vibe_core.data import DataVibe
from vibe_core.data.core_types import BaseVibe
from vibe_dev.testing.op_tester import OpTester

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "select_necessary_coverage_items.yaml"
)


def test_op():
    bounds = shpg.Point(10, 10).buffer(5)
    bounds = [bounds, shpa.translate(bounds, -6, 6)]
    geom = shpg.Point(10, 10).buffer(10)
    input_geoms = [
        shpa.translate(geom, -7, 0),
        shpa.translate(geom, 8, 0),
        shpa.translate(geom, 0, 8),
        shpa.translate(geom, 5, 5),
    ]

    start_date = datetime(year=2021, month=7, day=10, tzinfo=timezone.utc)
    end_date = datetime(year=2021, month=7, day=28, tzinfo=timezone.utc)
    bounds_vibe = [DataVibe("bounds", (start_date, end_date), shpg.mapping(b), []) for b in bounds]
    input_vibe = [
        DataVibe(f"input{i}", (start_date, end_date), shpg.mapping(g), [])
        for i, g in enumerate(input_geoms)
    ]
    inputs = [bounds_vibe[:1], bounds_vibe[1:2], bounds_vibe]
    expected_out = [input_vibe[:2], [input_vibe[0], input_vibe[2]], input_vibe[:3]]

    for inp, out in zip(inputs, expected_out):
        output_vibe = OpTester(CONFIG_PATH).run(
            bounds_items=cast(BaseVibe, inp), items=cast(List[BaseVibe], input_vibe)
        )

        # Get op result
        output_name = "filtered_items"
        assert output_name in output_vibe
        items = output_vibe[output_name]
        assert isinstance(items, list)
        assert len(items) == len(out)
        assert items == out
