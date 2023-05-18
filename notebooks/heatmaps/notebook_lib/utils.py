import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
from geopandas import GeoSeries
from matplotlib import pyplot as plt
from shapely import geometry as shpg

from vibe_core.client import get_default_vibe_client
from vibe_core.data import ADMAgSeasonalFieldInput, ExternalReferenceList, Raster
from vibe_core.data.core_types import BaseGeometry, DataVibe


def view_output(archive_path: str, title: str):
    all = []
    column_name = "layer"
    match_str = "cluster(.*).shp"
    z = ZipFile(archive_path)
    output_shapes = [
        o
        for o in z.namelist()
        if o.endswith(".shp") and o != "cluster0.shp" and o != "cluster0.0.shp"
    ]

    for o in output_shapes:
        f = gpd.read_file(archive_path + "!" + o)
        f[column_name] = cast("re.Match[str]", re.search(match_str, o)).group(1)
        all.append(f)

    all = pd.concat(all)
    all[column_name] = all[column_name].astype(float)
    _, ax = plt.subplots(figsize=(7, 6))
    all.plot(ax=ax, column=column_name, legend=True, cmap="viridis")
    plt.axis("off")
    plt.title(title)


def create_heatmap(
    imagery: Raster, geojson_url: str, farm_boundary: str, parameters: Dict[str, Any]
) -> str:
    now = datetime.now()
    # create id
    geom_url_hash = str(hash(geojson_url))

    # read farm boundary
    data_frame = gpd.read_file(farm_boundary)
    geometry = shpg.mapping(data_frame["geometry"][0])

    # submit request to farmVibes cluster
    sample_inputs = ExternalReferenceList(
        id=geom_url_hash, time_range=(now, now), geometry=geometry, assets=[], urls=[geojson_url]
    )

    inputs = {"input_raster": imagery, "input_samples": sample_inputs}
    workflow = "farm_ai/agriculture/heatmap_sensor"
    name = "heatmap_example"

    out = submit_inputs_request(inputs, parameters, workflow, name)
    dv = cast(List[DataVibe], out["result"])[0]
    asset = dv.assets[0]
    return asset.path_or_url


def create_heatmap_admag(
    imagery: Raster, farm_infO: Dict[str, str], parameters: Dict[str, Any]
) -> str:
    sample_inputs = ADMAgSeasonalFieldInput(
        farmer_id=farm_infO["farmer_id"],
        seasonal_field_id=farm_infO["seasonal_field_id"],
        boundary_id=farm_infO["boundary_id"],
    )

    inputs = {"input_raster": imagery, "admag_input": sample_inputs}
    workflow = "farm_ai/agriculture/heatmap_sensor_admag"
    name = "heatmap_example"

    out = submit_inputs_request(inputs, parameters, workflow, name)
    dv = out["result"][0]
    asset = dv.assets[0]
    return asset.path_or_url


def submit_inputs_request(
    inputs: Union[Dict[str, Any], ExternalReferenceList],
    parameters: Dict[str, Any],
    workflow: str,
    name: str,
) -> Dict[str, Any]:
    client = get_default_vibe_client()

    run = client.run(
        workflow=workflow,
        name=name,
        input_data=inputs,
        parameters=parameters,
    )

    # display execution results
    run.monitor(refresh_time_s=5)

    if run.status == "done":
        assert run.output, "No output found in completed run"
        return run.output
    else:
        raise Exception(client.describe_run(run.id))


def get_raster_from_cluster(
    farm_boundary: str, time_range: Tuple[datetime, datetime], sr_id: int = 32611
) -> Optional[Raster]:
    # read farm boundary
    data_frame = gpd.read_file(farm_boundary)
    geometry = GeoSeries([data_frame["geometry"][0]], crs=sr_id).to_crs(4326)[0]
    geometry = shpg.mapping(geometry)

    return get_raster_from_cluster_using_geometry(geometry, time_range)


def get_raster_from_external(imagery_url: str, farm_boundary: str, sr_id: int = 32611) -> Raster:
    url_hash = str(hash(imagery_url))
    now = datetime.now()

    # read farm boundary
    data_frame = gpd.read_file(farm_boundary)
    geometry = GeoSeries([data_frame["geometry"][0]], crs=sr_id).to_crs(4326)[0]
    geometry = shpg.mapping(geometry)

    inputs = ExternalReferenceList(
        id=url_hash, time_range=(now, now), geometry=geometry, assets=[], urls=[imagery_url]
    )

    out = submit_inputs_request(
        inputs=inputs,
        parameters={},
        workflow="data_ingestion/user_data/ingest_raster",
        name="image_example",
    )

    return cast(List[Raster], out["raster"])[0]


def get_seasonal_field(farm_infO: Dict[str, str], parameters: Dict[str, Any]) -> Dict[str, Any]:
    sample_inputs = ADMAgSeasonalFieldInput(
        farmer_id=farm_infO["farmer_id"],
        seasonal_field_id=farm_infO["seasonal_field_id"],
        boundary_id=farm_infO["boundary_id"],
    )
    inputs = {"admag_input": sample_inputs}

    workflow = "data_ingestion/admag/admag_seasonal_field"
    name = "seasonal_field"

    out = submit_inputs_request(inputs, parameters, workflow, name)
    return out["seasonal_field"][0]


def get_raster_from_cluster_using_geometry(
    farm_boundary: Dict[str, Any], time_range: Tuple[datetime, datetime]
) -> Optional[Raster]:
    id = str(hash(time_range))
    obj = DataVibe(id=id, time_range=time_range, geometry=farm_boundary, assets=[])
    inputs = {"user_input": obj}
    out = submit_inputs_request(
        inputs=inputs,
        parameters={},
        workflow="data_ingestion/sentinel2/preprocess_s2",
        name="image_example",
    )
    return cast(List[Raster], out["raster"])[0]
