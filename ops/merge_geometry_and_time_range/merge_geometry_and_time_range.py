import hashlib
from typing import Dict

from vibe_core.data import DataVibe


def callback_builder():
    def callback(geometry: DataVibe, time_range: DataVibe) -> Dict[str, DataVibe]:
        id = hashlib.sha256(
            f"merge geometry and time range {geometry.id}{time_range.id}".encode()
        ).hexdigest()
        return {
            "merged": DataVibe(
                id=id, geometry=geometry.geometry, time_range=time_range.time_range, assets=[]
            )
        }

    return callback
