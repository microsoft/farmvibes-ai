import os
import time
from datetime import datetime
from typing import Any, Dict, Union, cast

import geopandas as gpd
import pytest
from shapely import geometry as shpg
from shapely.geometry import MultiPolygon, Polygon

from vibe_core.client import FarmvibesAiClient, get_default_vibe_client
from vibe_core.data import ADMAgSeasonalFieldInput, DataVibe, ExternalReferenceList
from vibe_core.data.core_types import GeometryCollection
from vibe_core.data.rasters import Raster
from vibe_dev.testing.op_tester import OpTester

FAKE_TIME_RANGE = (datetime(2022, 6, 30), datetime(2022, 7, 2))
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "soil_sample_heatmap_using_neighbors.yaml",
)


@pytest.fixture
def vibe_client():
    return get_default_vibe_client()


@pytest.fixture
def vibe_geometry_dict() -> Dict[str, Any]:
    farm_boundary = "op_resources/heatmap_sensor/long_block_boundary_4326.geojson"
    data_frame = gpd.read_file(farm_boundary)
    geometry = shpg.mapping(data_frame["geometry"][0])  # type: ignore
    return geometry


@pytest.fixture
def vibe_geometry_shapely() -> Union[MultiPolygon, Polygon]:
    farm_boundary = "op_resources/heatmap_sensor/long_block_boundary_4326.geojson"
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
        time_range=(datetime(2022, 6, 30), datetime(2022, 7, 2)),
    )

    while run.status == "running" or run.status == "pending":
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
def download_samples_cluster(
    vibe_client: FarmvibesAiClient, vibe_geometry_dict: Dict[str, Any]
) -> GeometryCollection:
    geojson_url = "<SAS URL>"
    url_hash = str(hash(geojson_url))
    now = datetime.now()

    inputs = ExternalReferenceList(
        id=url_hash,
        time_range=(now, now),
        geometry=vibe_geometry_dict,
        assets=[],
        urls=[geojson_url],
    )
    run = vibe_client.run(
        workflow="data_ingestion/user_data/ingest_geometry",
        name="geometry_example",
        input_data=inputs,
    )

    while run.status == "running" or run.status == "pending":
        continue
    time.sleep(5)
    if run.status == "done":
        obj: GeometryCollection = run.output["geometry"][0]  # type: ignore
        return obj

    raise RuntimeError("Download samples cluster request failed - ")


@pytest.fixture
def download_samples_boundary(
    vibe_client: FarmvibesAiClient, vibe_geometry_dict: Dict[str, Any]
) -> GeometryCollection:
    geojson_url = "<SAS URL>"
    url_hash = str(hash(geojson_url))
    now = datetime.now()

    inputs = ExternalReferenceList(
        id=url_hash,
        time_range=(now, now),
        geometry=vibe_geometry_dict,
        assets=[],
        urls=[geojson_url],
    )
    run = vibe_client.run(
        workflow="data_ingestion/user_data/ingest_geometry",
        name="geometry_example",
        input_data=inputs,
    )

    while run.status == "running" or run.status == "pending":
        continue
    time.sleep(5)
    if run.status == "done":
        obj: GeometryCollection = run.output["geometry"][0]  # type: ignore
        return obj

    raise RuntimeError("Download samples boundary request failed - ")


@pytest.fixture
def data_vibe(vibe_geometry_dict: Dict[str, Any]):
    id = str(hash("test_minimums_samples_heatmap"))
    return DataVibe(id, FAKE_TIME_RANGE, vibe_geometry_dict, [])


@pytest.mark.skip(reason="Dependent on the cluster")
def test_heatmap_c(
    download_sentinel_cluster: Raster,
    download_samples_cluster: GeometryCollection,
    download_samples_boundary: GeometryCollection,
):
    op_ = OpTester(CONFIG_PATH)
    parameters = {"attribute_name": "C", "simplify": "simplify", "tolerance": 1.0}
    op_.update_parameters(parameters)
    output_data = op_.run(
        raster=download_sentinel_cluster,
        samples=download_samples_cluster,
        samples_boundary=download_samples_boundary,
    )

    # Get op result
    assert "result" in output_data


@pytest.fixture
def prescriptions(vibe_client: FarmvibesAiClient):
    parameters = {
        "base_url": "base_url",
        "client_id": "client_id",
        "client_secret": "client_secret",
        "authority": "authority",
        "default_scope": "default_scope",
    }
    sample_inputs = ADMAgSeasonalFieldInput(
        party_id="a460c833-7b96-4905-92ed-f19800b87185",
        seasonal_field_id="7db1a756-b898-4ecb-8608-bc2476f242a9",
    )
    inputs = {"admag_input": sample_inputs}
    run = vibe_client.run(
        workflow="data_ingestion/admag/prescriptions",
        name="prescriptions_example",
        input_data=inputs,  # type: ignore
        parameters=parameters,
    )

    while run.status == "running" or run.status == "pending":
        continue

    if run.status == "done":
        obj = cast(GeometryCollection, run.output["response"][0])  # type: ignore
        return obj
    raise RuntimeError("Fetch prescriptions failed - ")
