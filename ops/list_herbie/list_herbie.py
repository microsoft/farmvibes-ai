# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from herbie import Herbie_latest

from vibe_core.data import DataVibe, HerbieProduct

N = 6  # latest file within the last N*frequecy hours


class CallbackBuilder:
    def __init__(
        self,
        model: str,
        product: str,
        frequency: int,
        search_text: str,
        forecast_lead_times: Optional[List[int]] = None,
        forecast_start_date: Optional[str] = None,
    ):
        if forecast_lead_times is not None and forecast_start_date is not None:
            raise ValueError(
                "You cannot specify 'forecast_lead_times' and"
                " 'forecast_start_date' at the same time."
            )
        self.model = model
        self.product = product
        self.frequency = frequency
        self.forecast_lead_times = forecast_lead_times
        self.search_text = search_text
        self.forecast_start_date = forecast_start_date

    def _get_list(self, input_item: DataVibe):
        start = input_item.time_range[0].replace(tzinfo=None)
        end = input_item.time_range[1].replace(tzinfo=None)
        if self.forecast_lead_times is None:
            if self.forecast_start_date is None:
                H = Herbie_latest(n=N, freq=f"{self.frequency}H", model=self.model)
                latest = H.date.to_pydatetime()
            else:
                latest = datetime.strptime(self.forecast_start_date, "%Y-%m-%d %H:%M")
            if end > latest or self.forecast_start_date is not None:
                plist = [(t, 0) for t in pd.date_range(start, latest, freq=f"{self.frequency}H")]
                r = len(pd.date_range(start, end, freq=f"{self.frequency}H"))
                last = plist[-1][0]
                plist += [
                    (last, int(lead))
                    for lead in (np.arange(1, r - len(plist) + 1) * self.frequency)
                ]
            else:
                plist = [(t, 0) for t in pd.date_range(start, end, freq=f"{self.frequency}H")]
        else:
            plist = [
                (t, lead)
                for t in pd.date_range(start, end, freq=f"{self.frequency}H")
                for lead in range(
                    self.forecast_lead_times[0],
                    self.forecast_lead_times[1],
                    self.forecast_lead_times[2],
                )
            ]

        return plist

    def __call__(self):
        def list_herbie(
            input_item: DataVibe,
        ) -> Dict[str, List[HerbieProduct]]:
            plist = self._get_list(input_item)

            products = [
                HerbieProduct.clone_from(
                    input_item,
                    hashlib.sha256(
                        (
                            f"{self.model}-{self.product}-"
                            f"{lead}-{self.search_text}-"
                            f"{str(input_item.geometry)}-{str(t)}"
                        ).encode()
                    ).hexdigest(),
                    assets=[],
                    time_range=(
                        t.tz_localize(input_item.time_range[0].tzinfo),
                        t.tz_localize(input_item.time_range[0].tzinfo),
                    ),
                    model=self.model,
                    product=self.product,
                    lead_time_hours=lead,
                    search_text=self.search_text,
                )
                for t, lead in plist
            ]
            return {"product": products}

        return list_herbie
