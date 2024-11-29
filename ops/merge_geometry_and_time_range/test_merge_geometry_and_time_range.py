# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime

from shapely import geometry as shpg

from vibe_core.data import DataVibe
from vibe_dev.testing.op_tester import OpTester

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "merge_geometry_and_time_range.yaml"
)


def test_op():
    vibe1 = DataVibe(
        id="1",
        geometry=shpg.mapping(shpg.box(0, 0, 1, 1)),
        time_range=(datetime(2020, 1, 1), datetime(2020, 2, 2)),
        assets=[],
    )
    vibe2 = DataVibe(
        id="2",
        geometry=shpg.mapping(shpg.box(0, 0, 2, 2)),
        time_range=(datetime(2021, 1, 1), datetime(2021, 2, 2)),
        assets=[],
    )
    op_tester = OpTester(CONFIG_PATH)
    out = op_tester.run(geometry=vibe1, time_range=vibe2)
    assert "merged" in out
    out_vibe = out["merged"]
    assert isinstance(out_vibe, DataVibe)
    assert out_vibe.geometry == vibe1.geometry
    assert out_vibe.time_range == vibe2.time_range

    out = op_tester.run(geometry=vibe2, time_range=vibe1)
    assert "merged" in out
    out_vibe = out["merged"]
    assert isinstance(out_vibe, DataVibe)
    assert out_vibe.geometry == vibe2.geometry
    assert out_vibe.time_range == vibe1.time_range
