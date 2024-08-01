# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Dict, List

import pandas as pd
from shapely import geometry as shpg

from vibe_core.data import AssetVibe, FoodVibe, ProteinSequence, gen_guid


def append_nones(length: int, list_: List[str]):
    """
    Appends Nones to list to get length of list equal to `length`.
    If list is too long raise AttributeError
    """
    diff_len = length - len(list_)
    if diff_len < 0:
        raise AttributeError("Length error list is too long.")
    return list_ + [" 0"] * diff_len


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def protein_sequence_callback(
            food_item: FoodVibe,
        ) -> Dict[str, ProteinSequence]:
            protein_list = append_nones(3, food_item.fasta_sequence)

            guid = gen_guid()
            filepath = os.path.join(self.tmp_dir.name, f"{guid}.csv")

            df = pd.DataFrame(protein_list, columns=["protein_list"])
            df.to_csv(filepath, index=False)

            protein_sequence = ProteinSequence(
                gen_guid(),
                time_range=(datetime.now(), datetime.now()),  # these are just placeholders
                geometry=shpg.mapping(shpg.Point(0, 0)),  # this location is a placeholder
                assets=[AssetVibe(reference=filepath, type="text/csv", id=guid)],
            )

            return {"protein_sequence": protein_sequence}

        return protein_sequence_callback
