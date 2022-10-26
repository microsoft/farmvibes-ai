from typing import Any, Dict, List, TypeVar, Union

from vibe_core.data.core_types import OpIOType

T = TypeVar("T")


def ensure_list(input: Union[List[T], T]) -> List[T]:
    if isinstance(input, list):
        return input
    return [input]


def get_input_ids(input: OpIOType) -> Dict[str, Union[str, List[str]]]:
    return {
        k: [vv.get("id", "NO-ID") for vv in v] if isinstance(v, list) else v.get("id", "NO-ID")
        for k, v in input.items()
    }


def rename_keys(x: Dict[str, Any], key_dict: Dict[str, str]):
    renamed = x.copy()
    for old_key, new_key in key_dict.items():
        if old_key in x:
            renamed[new_key] = x[old_key]
            del renamed[old_key]
    return renamed


def format_double_escaped(s: str):
    """
    Encodes and decodes a double escaped input string.
    Useful for formating status/reason strings of VibeWorkflowRun
    """
    return s.encode("raw_unicode_escape").decode("unicode-escape")
