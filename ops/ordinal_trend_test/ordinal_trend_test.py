import os
from datetime import datetime as dt
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from scipy.stats import norm

from vibe_core.data import AssetVibe, OrdinalTrendTest, RasterPixelCount, gen_guid

NODATA = None
DATE_FORMAT = "%Y/%m/%d"


def cochran_armitage_trend_test(contingency_table: NDArray[Any]) -> Tuple[float, float]:
    contingency_table = np.array(contingency_table)

    row_sums = np.sum(contingency_table, axis=1)
    column_sums = np.sum(contingency_table, axis=0)
    total = np.sum(row_sums)

    row_weights = np.arange(contingency_table.shape[0])
    column_weights = np.arange(contingency_table.shape[1])

    # Expected value
    col_inner = np.inner(column_weights, column_sums)
    row_inner = np.inner(row_weights, row_sums)
    expected = col_inner * row_inner / total

    # Statistics
    statistic = np.inner(row_weights, np.inner(contingency_table, column_weights))

    # Theorical background can be found here:
    # https://real-statistics.com/chi-square-and-f-distributions/cochran-armitage-test/
    # https://doi.org/10.1002/0471249688.ch5
    variance_numerator = np.inner(row_weights**2, row_sums) - row_inner**2 / total
    variance_numerator *= np.inner(column_weights**2, column_sums) - col_inner**2 / total
    variance = variance_numerator / (total - 1)

    z_score = (statistic - expected) / np.sqrt(variance)
    p_value = 2 * norm.cdf(-np.abs(z_score))

    return float(p_value), float(z_score)


def load_contingency_table(pixel_counts: List[RasterPixelCount]) -> pd.DataFrame:
    columns = []
    for pixel_count in pixel_counts:
        columns.append(np.loadtxt(pixel_count.assets[0].path_or_url, delimiter=",", skiprows=1))

    # Return the unique values for the existing pixels
    unique_values = np.unique(np.concatenate(columns, axis=0)[:, 0])
    contingency_table = pd.DataFrame(index=unique_values)

    for pixel_count, column in zip(pixel_counts, columns):
        contingency_table[pixel_count.id] = pd.Series(column[:, 1], index=column[:, 0])

    return contingency_table.fillna(0)


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def callback(pixel_count: List[RasterPixelCount]) -> Dict[str, OrdinalTrendTest]:
            if len(pixel_count) < 2:
                raise ValueError("Ordinal trend test requires at least pixel count from 2 rasters.")

            # Order the pixel counts by using the first date in time_range
            pixel_count = sorted(pixel_count, key=lambda x: x.time_range[0])

            time_ranges = [
                f"{dt.strftime(r.time_range[0], DATE_FORMAT)}-"
                f"{dt.strftime(r.time_range[1], DATE_FORMAT)}"
                for r in pixel_count
            ]

            # Calculate the min and max dates for the rasters
            min_date = min([r.time_range[0] for r in pixel_count])
            max_date = max([r.time_range[1] for r in pixel_count])

            contingency_table = load_contingency_table(pixel_count)
            p_value, z_score = cochran_armitage_trend_test(contingency_table.values)

            contingency_table.index.name = "category"
            contingency_table.columns = time_ranges  # type: ignore

            guid = gen_guid()
            filepath = os.path.join(self.tmp_dir.name, f"{guid}.csv")
            contingency_table.to_csv(filepath)

            ordinal_trend_result = OrdinalTrendTest(
                gen_guid(),
                time_range=(min_date, max_date),
                geometry=pixel_count[0].geometry,
                assets=[AssetVibe(reference=filepath, type="text/csv", id=guid)],
                p_value=p_value,
                z_score=z_score,
            )

            return {"ordinal_trend_result": ordinal_trend_result}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
