# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Sequence, TypeVar

from vibe_core.data import S2ProcessingLevel, Sentinel2Product

T = TypeVar("T", bound=Sentinel2Product)

QUANTIFICATION_VALUE = 10000
SPACEEYE_TO_SPYNDEX_BAND_NAMES: Dict[str, str] = {
    "B02": "B",
    "B03": "G",
    "B04": "R",
    "B05": "RE1",
    "B06": "RE2",
    "B07": "RE3",
    "B08": "N",
    "B8A": "N2",
    "B11": "S1",
    "B12": "S2",
}


def find_s2_product(product_name: str, products: List[T]) -> T:
    for product in products:
        if product.product_name == product_name:
            return product
    raise ValueError(f"Could not find product with product name {product_name}.")


def verify_processing_level(
    items: Sequence[Sentinel2Product], processing_level: S2ProcessingLevel, prefix: str = ""
):
    invalid = set(
        [item.processing_level for item in items if item.processing_level != processing_level]
    )
    if invalid:
        raise ValueError(
            f"{prefix} {'e' if prefix else 'E'}xpected items with processing level "
            f"{processing_level}. Found items with processing level: {','.join(invalid)}"
        )
