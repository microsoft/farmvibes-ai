# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Dict, List, Union, cast

import pandas as pd
from protlearn.features import aaindex1
from shapely import geometry as shpg

from vibe_core.data import AssetVibe, FoodFeatures, FoodVibe, ProteinSequence, gen_guid

PROTLEARN_FEAT_LIST: List[str] = [
    "JOND750102_2nd",
    "GEOR030105_1st",
    "JOND920102_2nd",
    "HOPA770101_1st",
    "WERD780102_2nd",
    "FUKS010109_1st",
]

NUTRITIONAL_INFORMATION: List[str] = [
    "Dietary Fiber",
    "Magnesium",
    "Potassium",
    "Manganese",
    "Zinc",
    "Iron",
    "Copper",
    "Protein",
    "TRP",
    "THR",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "CYS",
    "PHE",
    "TYR",
    "VAL",
    "ARG",
    "HIS",
]

PROTEIN_INFORMATION: List[str] = ["1st family", "2nd family", "3rd family", "Food group"]

FOOD_GROUP_ID: Dict[str, int] = {
    "Cereal & cereal products": 1,
    "Roots & tubers": 2,
    "Legumes & oilseeds": 3,
    "Oil byproducts": 4,
    "Fish & fish products": 5,
    "Animal products": 6,
    "Milk products": 7,
    "Fruits & vegetable products": 8,
    "Others": 9,
    "Plant based ": 10,
    "Mixed food (animal + cereal product)": 11,
    "Mixed food (plant based)": 12,
    "Mixed food (cereal + legume)": 13,
    "Mixed food (cereal + animal product)": 14,
}

PROTEIN_FAMILY_ID: Dict[str, int] = {
    "": 0,
    "GLOBULIN": 1,
    "ALBUMIN": 2,
    "ALBUMINS": 2,
    "OVALBUMIN": 3,
    "OVOTRANSFERRIN": 4,
    "OVOMUCOID": 5,
    "CASEIN": 6,
    "GLYCININ": 7,
    "CONGLYCININ": 8,
    "GLUTELIN": 9,
    "GLIADINS": 10,
    "ZEIN": 11,
    "PROLAMIN": 12,
    "MYOSIN": 13,
    "MYOGLOBIN": 14,
    "PATATIN": 15,
    "LECTIN": 16,
    "LEGUMIN": 17,
    "OTHER": 18,
}


def encode_str(id_dict: Dict[str, int], val: Union[str, str]):
    if not val.strip():
        return 0

    try:
        encoded_id = id_dict[val]
    except KeyError:
        encoded_id = 18

    return encoded_id


def filter_protlearn_shap(protlearn_feats: pd.DataFrame):
    return protlearn_feats.filter(PROTLEARN_FEAT_LIST)


def extracting_protlearn(aminoacids1: str, aminoacids2: str, aminoacids3: str):
    """
    Reads in the aminoacid sequences from the fasta files
    Returns a dataframe with the Aaindex features obtained using protlearn package
    """
    aminoacids1 = aminoacids1[aminoacids1.rindex(" ") + 1 :]

    aaind1, inds1 = aaindex1(aminoacids1, standardize="zscore")  # type: ignore
    first = pd.DataFrame(aaind1, columns=inds1)  # type: ignore
    first = first.add_suffix("_1st")
    aminoacids2 = aminoacids2[aminoacids2.rindex(" ") + 1 :]

    try:
        aaind2, inds2 = aaindex1(aminoacids2, standardize="zscore")  # type: ignore
    except ValueError:
        aaind2 = 0
    second = pd.DataFrame(aaind2, index=range(1), columns=inds1)  # type: ignore
    second = second.add_suffix("_2nd")
    aminoacids3 = aminoacids3[aminoacids3.rindex(" ") + 1 :]

    try:
        aaind3, indes3 = aaindex1(aminoacids3, standardize="zscore")  # type: ignore
    except ValueError:
        aaind3 = 0
    third = pd.DataFrame(aaind3, index=range(1), columns=inds1)  # type: ignore
    third = third.add_suffix("_3rd")
    aaindex_feats = pd.concat([first, second, third], axis=1)
    return aaindex_feats


def read_protein(protein_df: pd.DataFrame):
    protein_list = protein_df["protein_list"]
    assert protein_list is not None, "Protein list column is missing"

    fasta_sequence0 = str(protein_list[0])

    try:
        fasta_sequence1 = str(protein_list[1])
    except KeyError:
        fasta_sequence1 = " "

    try:
        fasta_sequence2 = str(protein_list[2])
    except KeyError:
        fasta_sequence2 = " "

    return fasta_sequence0, fasta_sequence1, fasta_sequence2


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def protlearn_callback(
            food_item: FoodVibe, protein_sequence: ProteinSequence
        ) -> Dict[str, FoodFeatures]:
            protein_df = cast(
                pd.DataFrame, pd.read_csv(protein_sequence.assets[0].path_or_url, index_col=0)
            ).reset_index()

            fasta_sequence0, fasta_sequence1, fasta_sequence2 = read_protein(protein_df)

            aaindex_feats = extracting_protlearn(
                fasta_sequence0,
                fasta_sequence1,
                fasta_sequence2,
            )

            nutritional_data = [
                food_item.dietary_fiber,
                food_item.magnesium,
                food_item.potassium,
                food_item.manganese,
                food_item.zinc,
                food_item.iron,
                food_item.copper,
                food_item.protein,
                food_item.trp,
                food_item.thr,
                food_item.ile,
                food_item.leu,
                food_item.lys,
                food_item.met,
                food_item.cys,
                food_item.phe,
                food_item.tyr,
                food_item.val,
                food_item.arg,
                food_item.his,
            ]

            protein_family_food_type = [
                encode_str(PROTEIN_FAMILY_ID, food_item.protein_families[0]),
                encode_str(PROTEIN_FAMILY_ID, food_item.protein_families[1]),
                encode_str(PROTEIN_FAMILY_ID, food_item.protein_families[2]),
                encode_str(FOOD_GROUP_ID, food_item.food_group),
            ]

            nutritional_data_df = pd.DataFrame(nutritional_data, index=NUTRITIONAL_INFORMATION)
            protein_family_df = pd.DataFrame(protein_family_food_type, index=PROTEIN_INFORMATION)

            protlearn_df = filter_protlearn_shap(aaindex_feats)

            df = pd.concat([nutritional_data_df.T, protlearn_df, protein_family_df.T], axis=1)

            guid = gen_guid()
            filepath = os.path.join(self.tmp_dir.name, f"{guid}.csv")
            df.to_csv(filepath, index=False)

            food_features = FoodFeatures(
                gen_guid(),
                time_range=(datetime.now(), datetime.now()),  # these are just placeholders
                geometry=shpg.mapping(shpg.Point(0, 0)),  # this location is a placeholder
                assets=[AssetVibe(reference=filepath, type="text/csv", id=guid)],
            )

            return {"food_features": food_features}

        return protlearn_callback

    def __del__(self):
        self.tmp_dir.cleanup()
