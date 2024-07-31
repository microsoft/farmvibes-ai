from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from IPython.display import clear_output
from shapely.geometry import Point

from vibe_core.client import FarmvibesAiClient, get_default_vibe_client
from vibe_core.datamodel import RunConfig, RunConfigUser, SpatioTemporalJson


class Forecast:
    def __init__(
        self,
        workflow_name: str,
        geometry: Point,
        time_range: Tuple[datetime, datetime],
        parameters: List[Dict[str, str]],
        date_column: str = "date",
    ):
        self.client: FarmvibesAiClient = get_default_vibe_client()
        self.workflow_name = workflow_name
        self.geometry = geometry
        self.parameters = parameters
        self.time_range = time_range
        self.date_column = date_column

    def submit_download_request(self):
        """
        Submit request to worker to download forecast data
        """
        run_metadata_list = []
        runs = []
        for parameter in self.parameters:
            run_name = f"forecast_{parameter['weather_type']}"
            run = self.client.run(
                workflow=self.workflow_name,
                name=run_name,
                geometry=self.geometry,
                time_range=self.time_range,
                parameters=parameter,
            )

            run_metadata_list.append(
                {
                    "id": run.id,
                    "weather_type": parameter["weather_type"],
                }
            )
            runs.append(run)

        self.client.monitor(runs, 5)

        return run_metadata_list

    def get_run_status(self, run_list: List[Dict[str, str]]):
        clear_output(wait=True)
        out = []
        for run_item in run_list:
            o = self.client.describe_run(run_item["id"])
            print(f"Execution status for {run_item['weather_type']}: {o.details.status}")

            if o.details.status == "done":
                out.append(o)
            else:
                raise Exception(
                    f"Execution status for {run_item['weather_type']}: {o.details.status}"
                )

        return out

    def get_all_assets(self, details: RunConfigUser):
        asset_files = []
        output = details.output["weather_forecast"]
        record: Dict[str, Any]
        for record in cast(List[Dict[str, Any]], output):
            for value in record["assets"].values():
                asset_files.append(value["href"])
        df_assets = [pd.read_csv(f, index_col=False) for f in asset_files]
        df_out = pd.concat(df_assets)
        df_out = self.clean_forecast_data(forecast_df=df_out, run_details=details)
        return df_out

    def get_downloaded_data(self, run_list: List[Dict[str, str]], offset_hours: int = 0):
        """
        check the download status. If status is done, fetch the downloaded data
        """
        forecast_dataset = pd.DataFrame()
        out = self.get_run_status(run_list)
        for detail in out:
            df = self.get_all_assets(detail)

            # Offset from UTC to specified timezone
            df.index = df.index + pd.offsets.Hour(offset_hours)

            if not df.empty:
                forecast_dataset = pd.concat([forecast_dataset, df], axis=1)

        return forecast_dataset

    def clean_forecast_data(
        self,
        forecast_df: pd.DataFrame,
        run_details: RunConfig,
    ):
        df = forecast_df[self.date_column]
        assert isinstance(run_details.user_input, SpatioTemporalJson)
        start_date: datetime = run_details.user_input.start_date
        end_date: datetime = run_details.user_input.end_date

        # derive forecast data
        forecast_df.drop(columns=[self.date_column], inplace=True)
        a = forecast_df.values.tolist()
        o = pd.DataFrame([a])
        o = o.T

        df_date = pd.DataFrame(
            data=pd.date_range(start_date, end_date + timedelta(days=1), freq="h"),
            columns=[self.date_column],
        )

        # derive hours
        hours = [f"{str(i)}:00:00" for i in range(24)]
        list_hours = [hours for _ in range(forecast_df.shape[0])]

        assert run_details.parameters is not None, "Parameters are not defined"
        # transform forecast data with date and time
        df = pd.DataFrame(
            data={
                self.date_column: df.values,
                "time": list_hours,
                run_details.parameters["weather_type"]: o[0],
            }
        )
        df = df.explode(column=["time", run_details.parameters["weather_type"]])
        df[self.date_column] = df[self.date_column].astype(str) + " " + df["time"]
        df[self.date_column] = pd.to_datetime(df[self.date_column].values)

        df.drop(columns=["time"], inplace=True)
        df = pd.merge(df_date, df, how="left", left_on=self.date_column, right_on=self.date_column)

        df.reset_index()
        df.set_index(self.date_column, inplace=True)
        df.sort_index(ascending=True, inplace=True)
        df[run_details.parameters["weather_type"]] = df[
            run_details.parameters["weather_type"]
        ].values.astype(np.float32)

        # rename columns with suffix forecast
        df.rename(
            columns={
                run_details.parameters[
                    "weather_type"
                ]: f"{run_details.parameters['weather_type']}_forecast"
            },
            inplace=True,
        )

        # interpolate to derive missing data
        df = df.interpolate(method="from_derivatives")
        assert df is not None, "Interpolation deleted all data"
        df = df.dropna()
        return df
