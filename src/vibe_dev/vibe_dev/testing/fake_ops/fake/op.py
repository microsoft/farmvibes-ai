# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime
from typing import Any

from vibe_core.data import DataVibe


def print_args(user_data: Any):
    try:
        now = datetime.now()
        user_data.data = "Processed " + user_data.data
        print(user_data.data)
        return {
            "processed_data": [
                DataVibe(
                    user_data.data,
                    (now, now),
                    {
                        "type": "Point",
                        "coordinates": [0.0, 0.0],
                        "properties": {"name": user_data.data},
                    },
                    [],
                )
            ]
        }
    except Exception:
        return {"processed_data": user_data}


def callback_builder(**kw: Any):
    return print_args
