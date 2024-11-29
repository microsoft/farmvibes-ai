# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Dict, List, cast

import pandas as pd

from vibe_core.data import AssetVibe, DataSummaryStatistics, TimeSeries, gen_guid


class CallbackBuilder:
    def __init__(self, masked_thr: float):
        self.tmp_dir = TemporaryDirectory()
        self.masked_thr = masked_thr

    def __call__(self):
        def callback(stats: List[DataSummaryStatistics]) -> Dict[str, List[TimeSeries]]:
            df = pd.concat(
                cast(
                    List[pd.DataFrame],
                    [
                        pd.read_csv(s.assets[0].url, index_col="date", parse_dates=True)
                        for s in stats
                    ],
                )
            )
            assert df is not None, "DataFrame is None, that should not happen"
            # Filter out items above threshold
            df = cast(pd.DataFrame, df[df["masked_ratio"] <= self.masked_thr])  # type: ignore
            if df.empty:
                raise RuntimeError(
                    f"No available data with less than {self.masked_thr:.1%} masked data"
                )
            df.sort_index(inplace=True)
            guid = gen_guid()
            filepath = os.path.join(self.tmp_dir.name, f"{guid}.csv")
            df.to_csv(filepath)
            min_date = df.index.min().to_pydatetime()  # type: ignore
            max_date = df.index.max().to_pydatetime()  # type: ignore
            timeseries = TimeSeries(
                gen_guid(),
                time_range=(min_date, max_date),  # type: ignore
                geometry=stats[0].geometry,
                assets=[AssetVibe(reference=filepath, type="text/csv", id=guid)],
            )

            return {"timeseries": [timeseries]}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
