from dataclasses import fields
from typing import List

import numpy as np
import rasterio
from shapely import geometry as shpg

from vibe_core.data import DataVibe

IGNORE_FIELDS = ["id", "assets"]
RTOL = 1e-5
ATOL = 1e-8
SHAPE_TOL = 1e-6


def assert_all_fields_close(
    x: DataVibe, y: DataVibe, ignore_fields: List[str] = IGNORE_FIELDS
) -> None:
    compare_fields = [f.name for f in fields(x) if f.name not in ignore_fields]
    for f in compare_fields:
        x_f = getattr(x, f)
        y_f = getattr(y, f)
        if f == "geometry":
            # Option 1: Check if they are within each other with some tolerance
            # (x.within(y.buffer(1e-6)) & x.buffer(1e-6).contains(y))
            # Option 2: Check per point equivalence with a tolerance
            assert shpg.shape(x_f).equals_exact(
                shpg.shape(y_f), SHAPE_TOL
            ), f"Geometries are not equal with tolerance {SHAPE_TOL}"
        else:
            assert x_f == y_f, f"Field {f} is different: {x_f} != {y_f}"


def assert_all_close(x: DataVibe, y: DataVibe) -> None:
    assert type(x) == type(y), f"Data types are different: {type(x)} != {type(y)}"
    assert_all_fields_close(x, y)
    for a1, a2 in zip(x.assets, y.assets):
        assert a1.type == a2.type, f"Assets have different mimetypes: {a1.type} != {a2.type}"
        if a1.type == "image/tiff":
            with rasterio.open(a1.url) as src1:
                with rasterio.open(a2.url) as src2:
                    assert src1.meta == src2.meta, "TIFF files have different metadata"
                    ar1 = src1.read()
                    ar2 = src2.read()
                    assert np.allclose(
                        ar1, ar2, rtol=RTOL, atol=ATOL, equal_nan=True
                    ), f"Raster values are not all close with rtol={RTOL} and atol={ATOL}"
