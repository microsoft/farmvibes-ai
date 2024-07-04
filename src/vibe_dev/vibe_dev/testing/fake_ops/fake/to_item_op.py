from typing import Any, List

from vibe_core.data import DataVibe


def callback(user_data: List[DataVibe]):
    return {"processed_data": DataVibe.clone_from(user_data[0], id=user_data[0].id, assets=[])}


def callback_builder(**kw: Any):
    return callback
