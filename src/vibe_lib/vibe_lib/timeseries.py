import mimetypes
import os

import pandas as pd

from vibe_core.data import AssetVibe, gen_guid


def save_timeseries_to_asset(timeseries: pd.DataFrame, output_dir: str) -> AssetVibe:
    """
    Save dataframe to CSV file and return corresponding asset
    """
    out_id = gen_guid()
    filepath = os.path.join(output_dir, f"{out_id}.csv")
    timeseries.to_csv(filepath)
    new_asset = AssetVibe(reference=filepath, type=mimetypes.types_map[".csv"], id=out_id)
    return new_asset
