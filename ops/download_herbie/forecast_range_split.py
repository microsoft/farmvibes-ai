# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List

import pandas as pd

from vibe_core.data import DataVibe, gen_hash_id


class CallbackBuilder:
    def __init__(
        self,
        forecast_lead_times: List[int],
        weather_type: str,
    ):
        self.weather_type = weather_type
        self.frequency = forecast_lead_times[1] - forecast_lead_times[0]

    def get_forecast_weather(self, user_input: DataVibe) -> List[DataVibe]:
        dates = pd.date_range(
            user_input.time_range[0], user_input.time_range[1], freq=f"{str(self.frequency)}H"
        )

        forecasts = [
            DataVibe(
                gen_hash_id(
                    name=self.weather_type,
                    geometry=user_input.geometry,
                    time_range=(date, date),
                ),
                (date, date),
                user_input.geometry,
                [],
            )
            for date in dates
        ]

        return forecasts

    def __call__(self):
        def range_split_initialize(user_input: List[DataVibe]) -> Dict[str, List[DataVibe]]:
            download_period = self.get_forecast_weather(user_input[0])
            return {"download_period": download_period}

        return range_split_initialize
