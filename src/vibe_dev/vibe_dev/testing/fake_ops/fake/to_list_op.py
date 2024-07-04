from typing import Any

from vibe_core.data import DataVibe


def callback_builder(**kw: Any):
    num_items = kw.get("num_items", 1)

    def callback(user_data: DataVibe):
        return {
            "processed_data": [
                DataVibe.clone_from(user_data, id=f"{user_data.id}_{i}", assets=[])
                for i in range(num_items)
            ]
        }

    return callback
