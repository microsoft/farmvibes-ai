# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from typing import Any, Dict, Union, cast

import geopandas as gpd
import pytest
from shapely import geometry as shpg
from shapely.geometry import MultiPolygon, Polygon

from vibe_core.client import FarmvibesAiClient, get_default_vibe_client
from vibe_core.data import ADMAgSeasonalFieldInput, ExternalReferenceList
from vibe_core.data.core_types import GeometryCollection
from vibe_core.data.rasters import Raster
from vibe_dev.testing.op_tester import OpTester

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "soil_sample_heatmap_using_classification.yaml",
)


@pytest.fixture
def vibe_client():
    return get_default_vibe_client()


@pytest.fixture
def vibe_geometry_dict() -> Dict[str, Any]:
    farm_boundary = "op_resources/heatmap_sensor/sensor_farm_boundary.geojson"
    data_frame = gpd.read_file(farm_boundary)
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
        time_range=(datetime(2022, 6, 30), datetime(2022, 7, 2)),
    )

    while run is None or run.status == "running" or run.status == "pending":
        continue

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

    if run.status == "done":
        obj: Raster = run.output["index_raster"][0]  # type: ignore
        return obj

    raise RuntimeError("Download Raster request failed")


@pytest.fixture
def download_samples_cluster(
    vibe_client: FarmvibesAiClient, vibe_geometry_dict: Dict[str, Any]
) -> GeometryCollection:
    geojson_url = "<SAS Url>"
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

    while run is None or run.status == "running" or run.status == "pending":
        continue

    if run.status == "done":
        obj: GeometryCollection = run.output["geometry"][0]  # type: ignore
        return obj

    raise RuntimeError("Download samples request failed - ")


@pytest.mark.skip(reason="Dependent on the cluster")
@pytest.mark.parametrize("index", ["evi"])
def test_heatmap_c(download_index_cluster: Raster, download_samples_cluster: GeometryCollection):
    op_ = OpTester(CONFIG_PATH)
    parameters = {
        "attribute_name": "C",
        "buffer": 3,
        "bins": 4,
        "simplify": "simplify",
        "tolerance": 1.0,
        "data_scale": False,
        "max_depth": 50,
        "n_estimators": 25,
        "random_state": 100,
    }
    op_.update_parameters(parameters)
    output_data = op_.run(raster=download_index_cluster, samples=download_samples_cluster)

    # Get op result
    assert "result" in output_data


@pytest.mark.skip(reason="Dependent on the cluster")
@pytest.mark.parametrize("index", ["evi"])
def test_heatmap_n(download_index_cluster: Raster, download_samples_cluster: GeometryCollection):
    op_ = OpTester(CONFIG_PATH)
    parameters = {
        "attribute_name": "N",
        "buffer": 10,
        "bins": 4,
        "simplify": "simplify",
        "tolerance": 1.0,
        "data_scale": True,
        "max_depth": 50,
        "n_estimators": 25,
        "random_state": 100,
    }
    op_.update_parameters(parameters)
    output_data = op_.run(raster=download_index_cluster, samples=download_samples_cluster)

    # Get op result
    assert "result" in output_data


@pytest.mark.skip(reason="Dependent on the cluster")
@pytest.mark.parametrize("index", ["pri"])
def test_heatmap_ph(download_index_cluster: Raster, download_samples_cluster: GeometryCollection):
    op_ = OpTester(CONFIG_PATH)
    parameters = {
        "attribute_name": "pH",
        "buffer": 10,
        "bins": 4,
        "simplify": "simplify",
        "tolerance": 1.0,
        "data_scale": False,
        "max_depth": 50,
        "n_estimators": 25,
        "random_state": 100,
    }
    op_.update_parameters(parameters)
    output_data = op_.run(raster=download_index_cluster, samples=download_samples_cluster)

    # Get op result
    assert "result" in output_data


@pytest.mark.skip(reason="Dependent on the cluster")
@pytest.mark.parametrize("index", ["evi"])
def test_heatmap_p(download_index_cluster: Raster, download_samples_cluster: GeometryCollection):
    parameters = {
        "attribute_name": "P",
        "buffer": 3,
        "bins": 4,
        "simplify": "simplify",
        "tolerance": 1.0,
        "data_scale": True,
        "max_depth": 50,
        "n_estimators": 25,
        "random_state": 100,
    }
    op_ = OpTester(CONFIG_PATH)
    op_.update_parameters(parameters)
    output_data = op_.run(raster=download_index_cluster, samples=download_samples_cluster)

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


@pytest.mark.skip(reason="Dependent on the cluster")
@pytest.mark.parametrize("index", ["evi"])
def test_heatmap_p_admag(download_index_cluster: Raster, prescriptions: GeometryCollection):
    parameters = {
        "attribute_name": "P",
        "buffer": 3,
        "bins": 4,
        "simplify": "simplify",
        "tolerance": 1.0,
        "data_scale": True,
        "max_depth": 50,
        "n_estimators": 25,
        "random_state": 100,
    }
    op_ = OpTester(CONFIG_PATH)
    op_.update_parameters(parameters)
    output_data = op_.run(raster=download_index_cluster, samples=prescriptions)

    # Get op result
    assert "result" in output_data
