import os
from datetime import datetime

from shapely import geometry as shpg

from vibe_core.data import DataVibe
from vibe_dev.testing.op_tester import OpTester

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_geometries.yaml")


def test_op():
    geoms = [shpg.box(0, 0, 1, 1), shpg.box(0, 0, 2, 2)]
    items = [
        DataVibe(
            id=f"{i}",
            geometry=shpg.mapping(g),
            time_range=(datetime.now(), datetime.now()),
            assets=[],
        )
        for i, g in enumerate(geoms)
    ]
    op_tester = OpTester(CONFIG_PATH)
    op_tester.update_parameters({"method": "union"})
    out = op_tester.run(items=items)  # type: ignore
    assert "merged" in out
    out_vibe = out["merged"]
    assert isinstance(out_vibe, DataVibe)
    assert shpg.shape(out_vibe.geometry).equals(geoms[-1])
    assert out_vibe.time_range == items[0].time_range

    op_tester.update_parameters({"method": "intersection"})
    out = op_tester.run(items=items)  # type: ignore
    assert "merged" in out
    out_vibe = out["merged"]
    assert isinstance(out_vibe, DataVibe)
    assert shpg.shape(out_vibe.geometry).equals(geoms[0])
    assert out_vibe.time_range == items[0].time_range
