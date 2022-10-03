import warnings
from typing import Any, Dict, Type, TypeVar, cast

GenericTypeVibe = TypeVar("GenericTypeVibe", bound=Type[Any])

__DATA_REGISTRY: Dict[str, Type[Any]] = {}


def register_vibe_datatype(classdef: GenericTypeVibe) -> GenericTypeVibe:
    id = get_name(classdef)
    if id in __DATA_REGISTRY:
        warnings.warn(f"Class {id} already registered.", DeprecationWarning, stacklevel=2)
        return cast(GenericTypeVibe, retrieve(id))
    __DATA_REGISTRY[id] = classdef

    return cast(GenericTypeVibe, classdef)


def retrieve(id: str) -> Type[Any]:
    return __DATA_REGISTRY[id]


def get_name(classdef: Type[Any]) -> str:
    return classdef.__name__


def get_id(classdef: Type[Any]) -> str:
    id = get_name(classdef)

    return id
