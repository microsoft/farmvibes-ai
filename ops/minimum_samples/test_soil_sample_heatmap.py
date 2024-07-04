import os
import time
from datetime import datetime
from typing import Any, Dict, Union, cast

import geopandas as gpd
import pytest
from shapely import geometry as shpg
from shapely.geometry import MultiPolygon, Polygon

from vibe_core.client import FarmvibesAiClient, get_default_vibe_client
from vibe_core.data import DataVibe
from vibe_core.data.rasters import Raster
from vibe_dev.testing.op_tester import OpTester

FAKE_TIME_RANGE = (datetime(2022, 6, 30), datetime(2022, 7, 2))
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "find_soil_sample_locations.yaml"
)


@pytest.fixture
def vibe_client():
    return get_default_vibe_client()


@pytest.fixture
def vibe_geometry_dict() -> Dict[str, Any]:
    farm_boundary = "op_resources/nutrients/long_block_boundary.geojson"
    data_frame = gpd.read_file(farm_boundary, crs="EPSG:32611").to_crs("EPSG:4326")  # type: ignore
    geometry = shpg.mapping(data_frame["geometry"][0])  # type: ignore
    return geometry


@pytest.fixture
def vibe_geometry_shapely() -> Union[MultiPolygon, Polygon]:
    farm_boundary = "op_resources/heatmap_sensor/sensor_farm_boundary.geojson"
    data_frame = gpd.read_file(farm_boundary)
    if not data_frame.empty:
        geometry = data_frame["geometry"][0]  # type: ignore
        return cast(MultiPolygon, geometry)

    raise RuntimeError("Geometry is None")


@pytest.fixture
def download_sentinel_cluster(
    vibe_client: FarmvibesAiClient, vibe_geometry_shapely: Union[MultiPolygon, Polygon]
) -> Raster:
    run = vibe_client.run(
        workflow="data_ingestion/sentinel2/preprocess_s2",
        name="sentinel2_example",
        geometry=vibe_geometry_shapely,
        time_range=FAKE_TIME_RANGE,
    )

    while run is None or run.status == "running" or run.status == "pending":
        continue
    time.sleep(5)
    if run.status == "done":
        obj: Raster = run.output["raster"][0]  # type: ignore
        return obj

    raise RuntimeError("Download Raster request failed")


@pytest.fixture
def download_index_cluster(
    vibe_client: FarmvibesAiClient, download_sentinel_cluster: Raster, index: str
) -> Raster:
    parameters = {"index": index}

    run = vibe_client.run(
        workflow="data_processing/index/index",
        name="EVI_example",
        input_data=download_sentinel_cluster,
        parameters=parameters,
    )

    while run.status == "running" or run.status == "pending":
        continue
    time.sleep(5)
    if run.status == "done":
        obj: Raster = run.output["index_raster"][0]  # type: ignore
        return obj

    raise RuntimeError("Download Raster request failed")


@pytest.fixture
def data_vibe(vibe_geometry_dict: Dict[str, Any]):
    id = str(hash("test_minimums_samples"))
    return DataVibe(id, FAKE_TIME_RANGE, vibe_geometry_dict, [])


@pytest.mark.skip(reason="Dependent on the cluster")
@pytest.mark.parametrize("index", ["evi"])
def test_minimum_samples(download_index_cluster: Raster, data_vibe: DataVibe):
    op_ = OpTester(CONFIG_PATH)
    parameters = {
        "n_clusters": 5,
        "sieve_size": 2,
    }
    op_.update_parameters(parameters)
    output_data = op_.run(raster=download_index_cluster, user_input=data_vibe)

    # Get op result
    assert "locations" in output_data
