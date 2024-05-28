from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from matplotlib import pyplot as plt
from shapely import geometry as shpg

from vibe_core.client import get_default_vibe_client
from vibe_core.data import ExternalReferenceList, Raster
from vibe_core.data.core_types import AssetVibe, DataVibe


def download_sentinel_raster(
    farm_boundary: str, time_range: Tuple[datetime, datetime], sr_id: int = 32611
) -> Optional[Raster]:
    # read farm boundary
    data_frame = gpd.read_file(farm_boundary)
    geometry = GeoSeries([data_frame["geometry"][0]], crs=sr_id).to_crs(4326)[0]
    geometry = shpg.mapping(geometry)

    return get_raster_from_cluster_using_geometry(geometry, time_range)


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


def plot_sample_locations(cluster_boundaries: AssetVibe, samples_locations: AssetVibe):
    s_df = gpd.read_file(samples_locations.path_or_url)
    c_df = gpd.read_file(cluster_boundaries.path_or_url)

    fig, ax = plt.subplots(figsize=(8, 7))
    c_df.plot(
        legend=True,
        scheme="quantiles",
        ax=ax,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
        },
        cmap="viridis",
    )
    s_df.plot(color="black", ax=ax)
    plt.axis("off")


def get_sample_locations(
    indices_raster: Raster,
    farm_boundary: str,
    time_range: Tuple[datetime, datetime],
    parameters: Dict[str, Any],
    sr_id: int = 32611,
    plot: bool = True,
) -> Tuple[AssetVibe, AssetVibe]:
    # read farm boundary
    farm_df = cast(GeoDataFrame, gpd.read_file(farm_boundary, crs=sr_id))
    geometry = GeoSeries([farm_df["geometry"][0]], crs=sr_id).to_crs(4326)[0]
    geometry = shpg.mapping(geometry)

    id = str(hash(time_range))
    user_input = DataVibe(id=id, time_range=time_range, geometry=geometry, assets=[])

    inputs = {"input_raster": indices_raster, "user_input": user_input}
    workflow = "farm_ai/sensor/optimal_locations"
    name = "sample_locations"

    out = submit_inputs_request(inputs, parameters, workflow, name)
    cluster_boundaries = out["result"][0].assets[0]
    samples_locations = out["result"][0].assets[1]

    if plot:
        plot_sample_locations(cluster_boundaries, samples_locations)

    return cluster_boundaries, samples_locations


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
