# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import asdict
from typing import Any, List, Union

from vibe_core.data.core_types import BaseVibe


def callback(user_data: Union[BaseVibe, List[BaseVibe]]):
    if isinstance(user_data, list):
        return {"processed_data": [d.__class__(**asdict(d)) for d in user_data]}
    return {"processed_data": user_data.__class__(**asdict(user_data))}


def callback_builder(**kw: Any):  # type: ignore
    return callback
