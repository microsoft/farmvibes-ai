from vibe_core.data.core_types import OpIOType
from vibe_core.data.utils import deserialize_stac, serialize_stac

from .storage import ItemDict


class OpIOConverter:
    @staticmethod
    def serialize_output(output: ItemDict) -> OpIOType:
        return {k: serialize_stac(v) for k, v in output.items()}

    @staticmethod
    def deserialize_input(input_items: OpIOType) -> ItemDict:
        return {k: deserialize_stac(v) for k, v in input_items.items()}
