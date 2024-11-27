# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List

from vibe_core.data.core_types import gen_guid
from vibe_core.data.sentinel import ListTileData, Sequence2Tile, TileSequenceData


def callback_builder():
    """Op that splits a list of multiple TileSequence back to a list of Rasters"""

    def split_sequences(
        sequences: List[TileSequenceData],
    ) -> Dict[str, ListTileData]:
        rasters = [
            Sequence2Tile[type(sequence)].clone_from(
                sequence,
                id=gen_guid(),
                assets=[asset],
                time_range=sequence.asset_time_range[asset.id],
            )
            for sequence in sequences
            for asset in sequence.get_ordered_assets()
        ]
        return {"rasters": rasters}

    return split_sequences
