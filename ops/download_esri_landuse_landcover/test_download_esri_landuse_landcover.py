import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from shapely.geometry import Polygon, box, mapping

from vibe_core.data import CategoricalRaster
from vibe_core.data.core_types import DataVibe
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.planetary_computer import EsriLandUseLandCoverCollection

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "download_esri_landuse_landcover.yaml"
)


@patch(
    "vibe_lib.planetary_computer.get_available_collections",
    return_value=[EsriLandUseLandCoverCollection.collection],
)
@patch.object(EsriLandUseLandCoverCollection, "query_by_id")
@patch.object(
    EsriLandUseLandCoverCollection,
    "download_item",
    return_value=["/tmp/test_esri_landuse_landcover.tif"],
)
def test_op(_: MagicMock, __: MagicMock, ___: MagicMock):
    latitude = 42.21422
    longitude = -93.22890
    buffer = 0.001
    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    polygon: Polygon = box(*bbox, ccw=True)
    start_date = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    end_date = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)

    input: DataVibe = DataVibe(
        id=str("47P-2017"),
        time_range=(
            start_date,
            end_date,
        ),
        geometry=mapping(polygon),  # type: ignore
        assets=[],
    )

    output_data = OpTester(CONFIG_PATH).run(**{"input_product": input})

    # Get op result
    output_name = "downloaded_product"
    assert output_name in output_data
    output_product = output_data[output_name]
    assert isinstance(output_product, CategoricalRaster)
