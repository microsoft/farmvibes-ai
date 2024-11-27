# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import shutil
import tempfile
import warnings
from datetime import datetime
from socket import error as SocketError
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from herbie import FastHerbie

from vibe_core.data import AssetVibe, DataVibe, gen_guid
from vibe_core.data.weather import WeatherVibe

warnings.filterwarnings("ignore")

INDEX_COLUMN = "date"


class CallbackBuilder:
    def __init__(
        self,
        model: str,
        overwrite: bool,
        product: str,
        forecast_lead_times: List[int],
        search_text: str,
        weather_type: str,
    ):
        self.temp_dir = TemporaryDirectory()
        self.model = model
        self.overwrite = overwrite
        self.product = product
        self.forecast_lead_times = range(
            forecast_lead_times[0], forecast_lead_times[1], forecast_lead_times[2]
        )
        self.frequency = forecast_lead_times[1] - forecast_lead_times[0]
        self.search_text = search_text
        self.weather_type = weather_type

    def ping_herbie_source(self, date: datetime, coordinates: Tuple[float, float]):
        # initialize temporary directory
        tmp_dir = tempfile.mkdtemp()
        out_ = np.empty(0)
        try:
            # download forecast data
            fh = FastHerbie(
                [date],
                model=self.model,
                product=self.product,
                fxx=self.forecast_lead_times,
                save_dir=tmp_dir,
                overwrite=self.overwrite,
            )
            fh.download(searchString=self.search_text)

            # filter records nearest to coordinates
            ds = fh.xarray(searchString=self.search_text)

            out_key = [key for key in ds.keys() if key != "gribfile_projection"]
            out_ = ds.herbie.nearest_points(coordinates)[out_key[0]].values[0]

            if len(out_) < self.frequency:
                out_ = np.empty(0)

            del ds
            del fh
        except EOFError:
            # This error raises due to missing data.
            # ignore this error to continue download.
            pass
        except SocketError:
            pass
        except Exception:
            raise

        finally:
            # clear temporary directory
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return out_

    def get_forecast_weather(self, user_input: DataVibe) -> WeatherVibe:
        start_date = user_input.time_range[0].replace(tzinfo=None)
        end_date = user_input.time_range[1].replace(tzinfo=None)
        coords = tuple(user_input.geometry["coordinates"])
        dates = pd.date_range(start_date, end_date, freq=f"{str(self.frequency)}H")

        forecasts = []
        for date in dates:
            out_ = self.ping_herbie_source(date=date, coordinates=coords)
            if len(out_) > 0:
                forecasts.append([date] + list(out_))

        df = pd.DataFrame(
            data=forecasts,
            columns=[INDEX_COLUMN] + [f"step {x}" for x in self.forecast_lead_times],
        )

        # df = self.clean_forecast_data(forecast_df=df, start_date=start_date, end_date=end_date)
        out_path = os.path.join(self.temp_dir.name, f"{self.weather_type}.csv")
        df.to_csv(out_path, index=False)
        asset = AssetVibe(reference=out_path, type="text/csv", id=gen_guid())
        return WeatherVibe(
            gen_guid(),
            user_input.time_range,
            user_input.geometry,
            [asset],
        )

    def __call__(self):
        def weather_initialize(user_input: DataVibe) -> Dict[str, WeatherVibe]:
            weather_forecast = self.get_forecast_weather(user_input)
            return {"weather_forecast": weather_forecast}

        return weather_initialize

    def __del__(self):
        self.temp_dir.cleanup()
