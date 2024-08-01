# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, List, Union

from vibe_core.data import DataVibe


def callback(user_data: Union[DataVibe, List[DataVibe]]):
    if isinstance(user_data, list):
        return {"processed_data": [DataVibe.clone_from(d, id=d.id, assets=[]) for d in user_data]}
    return {"processed_data": DataVibe.clone_from(user_data, id=user_data.id, assets=[])}


def callback_builder(**kw: Any):
    return callback
